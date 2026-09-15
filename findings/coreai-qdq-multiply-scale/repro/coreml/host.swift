import Foundation
import CoreML
import CoreAI
import CryptoKit
import Darwin

struct Config: Decodable {
    let runtime: String
    let model_path: String
    let compiled_path: String?
    let entrypoint: String
    let input_name: String
    let output_name: String?
    let input_path: String
    let control_path: String
    let shape: [Int]
    let output_shape: [Int]
    let seconds: Double
    let warm_seconds: Double
    let quiet_seconds: Double
    let compute_units: String?
}
struct Call: Codable { let phase:String; let start_ns:UInt64; let end_ns:UInt64; let output:String; let sha256:String }
enum Failure: Error { case invalid(String) }
@main struct Host {
    static func mono()->UInt64 { DispatchTime.now().uptimeNanoseconds }
    static func save(_ value:Any,_ url:URL)throws { try JSONSerialization.data(withJSONObject:value,options:[.sortedKeys]).write(to:url,options:.atomic) }
    static func sha(_ d:Data)->String { SHA256.hash(data:d).map{String(format:"%02x",$0)}.joined() }
    static func fp16(_ d:Data)->[Float16] { d.withUnsafeBytes{ p in (0..<d.count/2).map{Float16(bitPattern:p.loadUnaligned(fromByteOffset:$0*2,as:UInt16.self))} } }
    static func bytes(_ x:[Float16])->Data { x.withUnsafeBytes{Data($0)} }
    static func offset(_ index:Int,_ shape:[Int],_ strides:[Int])->Int {
        var remaining=index;var offset=0
        for axis in (0..<shape.count).reversed(){offset += (remaining % shape[axis])*strides[axis];remaining /= shape[axis]}
        return offset
    }
    static func makeML(_ d:Data,_ shape:[Int])throws->MLMultiArray {
        let a=try MLMultiArray(shape:shape.map(NSNumber.init),dataType:.float16)
        let values=fp16(d); let p=a.dataPointer.assumingMemoryBound(to:Float16.self)
        let st=a.strides.map{$0.intValue}
        for i in 0..<values.count {p[offset(i,shape,st)]=values[i]}
        return a
    }
    static func copyML(_ a:MLMultiArray,_ shape:[Int])throws->Data {
        guard a.dataType == .float16 && a.shape.map({$0.intValue})==shape else {throw Failure.invalid("output_descriptor")}
        let p=a.dataPointer.assumingMemoryBound(to:Float16.self);let st=a.strides.map{$0.intValue}
        var x=[Float16](repeating:0,count:shape.reduce(1,*))
        for i in 0..<x.count {x[i]=p[offset(i,shape,st)]}
        return bytes(x)
    }
    static func makeAI(_ d:Data,_ shape:[Int])->NDArray {
        var a=NDArray(shape:shape,scalarType:.float16,strides:[shape[1]*shape[2]*shape[3],shape[2]*shape[3],shape[3],1]);let values=fp16(d)
        a.mutableView(as:Float16.self).withUnsafeMutablePointer{p,_,_ in values.withUnsafeBufferPointer{p.update(from:$0.baseAddress!,count:values.count)}}
        return a
    }
    static func copyAI(_ a:NDArray,_ shape:[Int])throws->Data {
        guard a.scalarType == .float16 else {throw Failure.invalid("output_dtype")}
        return try a.view(as:Float16.self).withUnsafePointer{p,s,st in
            guard (0..<s.count).map({s[$0]})==shape else {throw Failure.invalid("output_shape")}
            var x=[Float16](repeating:0,count:shape.reduce(1,*))
            let strides=(0..<st.count).map{st[$0]}
            for i in 0..<x.count {x[i]=p[offset(i,shape,strides)]};return bytes(x)
        }
    }
    static func plan(_ plan:MLComputePlan,_ url:URL)throws {
        var rows=[[String:Any]]()
        func walk(_ b:MLModelStructure.Program.Block,_ path:String) {
            for (i,op) in b.operations.enumerated() {
                let u=plan.deviceUsage(for:op)
                rows.append(["operator":op.operatorName,"outputs":op.outputs.map(\.name),"path":"\(path)/\(i)","preferred":u?.preferred.description ?? "unknown","supported":u?.supported.map(\.description) ?? []])
                for (j,child) in op.blocks.enumerated(){walk(child,"\(path)/\(i)/\(j)")}
            }
        }
        if case .program(let p)=plan.modelStructure {for (name,fn) in p.functions {walk(fn.block,name)}}
        try save(rows,url)
    }
    static func main() async {
        var destination:URL?
        do {
            if CommandLine.arguments.count==2 && CommandLine.arguments[1]=="--self-test" {
                for shape in [[1,3,1,7],[1,4,8,8]] {
                    let d=bytes((0..<shape.reduce(1,*)).map{Float16(($0 % 31)-15)/8})
                    guard try copyML(makeML(d,shape),shape)==d && copyAI(makeAI(d,shape),shape)==d else {throw Failure.invalid("tensor_copy_self_test")}
                }
                print("rank-4 ML/AI tensor copy self-test passed; no model loaded");return
            }
            guard CommandLine.arguments.count==3 else {throw Failure.invalid("usage")}
            let cfg=try JSONDecoder().decode(Config.self,from:Data(contentsOf:URL(fileURLWithPath:CommandLine.arguments[1])))
            let out=URL(fileURLWithPath:CommandLine.arguments[2]);destination=out
            try FileManager.default.createDirectory(at:out,withIntermediateDirectories:true)
            guard cfg.shape.count==4 && cfg.output_shape.count==4 && cfg.shape.allSatisfy({$0>0}) && cfg.output_shape.allSatisfy({$0>0}) && cfg.seconds>=0 else {throw Failure.invalid("shape_or_time")}
            let input=try Data(contentsOf:URL(fileURLWithPath:cfg.input_path));let control=try Data(contentsOf:URL(fileURLWithPath:cfg.control_path))
            guard input.count==cfg.shape.reduce(1,*)*2 && control.count==input.count else {throw Failure.invalid("input_size")}
            let negative=bytes(fp16(control).map{Float16(bitPattern:$0.bitPattern ^ 0x8000)})
            let inputs=[control,Data(repeating:0,count:input.count),negative,control,input,input]
            try save(["pid":getpid(),"stage":"before_load","monotonic_ns":mono()],out.appendingPathComponent("ready.json"))
            while !FileManager.default.fileExists(atPath:out.appendingPathComponent("go").path){try await Task.sleep(for:.milliseconds(20))}
            let loadStart=mono()
            var predict:((Int) async throws -> (UInt64,UInt64))!
            var lastBytes:(()throws->Data)!
            if cfg.runtime=="coreml" {
                let compiled=URL(fileURLWithPath:cfg.compiled_path!)
                if !FileManager.default.fileExists(atPath:compiled.path) {
                    let temp=try await MLModel.compileModel(at:URL(fileURLWithPath:cfg.model_path))
                    try FileManager.default.moveItem(at:temp,to:compiled)
                }
                let c=MLModelConfiguration()
                switch cfg.compute_units ?? "cpuAndNeuralEngine" {
                case "cpuAndNeuralEngine": c.computeUnits = .cpuAndNeuralEngine
                case "cpuOnly": c.computeUnits = .cpuOnly
                case "cpuAndGPU": c.computeUnits = .cpuAndGPU
                default: throw Failure.invalid("compute_units")
                }
                let model=try MLModel(contentsOf:compiled,configuration:c)
                try await plan(MLComputePlan.load(contentsOf:compiled,configuration:c),out.appendingPathComponent("plan.json"))
                let providers=try inputs.map{try MLDictionaryFeatureProvider(dictionary:[cfg.input_name:MLFeatureValue(multiArray:makeML($0,cfg.shape))])}
                var last:MLMultiArray?
                predict={i in try autoreleasepool {
                    let a=mono();let result=try model.prediction(from:providers[i]);let b=mono()
                    guard let array=result.featureValue(for:cfg.output_name!)?.multiArrayValue else {throw Failure.invalid("missing_output")}
                    last=array;return(a,b)
                }}
                lastBytes={try copyML(last!,cfg.output_shape)}
            } else if cfg.runtime=="coreai" {
                let model=try await AIModel(contentsOf:URL(fileURLWithPath:cfg.model_path),options:SpecializationOptions(preferredComputeUnitKind:.neuralEngine))
                guard let fn=try model.loadFunction(named:cfg.entrypoint),fn.descriptor.inputNames==[cfg.input_name],fn.descriptor.outputNames.count==1 else {throw Failure.invalid("function_descriptor")}
                let name=fn.descriptor.outputNames[0];let arrays=inputs.map{makeAI($0,cfg.shape)};var last:NDArray?
                predict={i in
                    _=model
                    let a=mono();var result=try await fn.run(inputs:[cfg.input_name:arrays[i]]);let b=mono()
                    guard let array=result.remove(name)?.ndArray else {throw Failure.invalid("missing_output")}
                    last=array;return(a,b)
                }
                lastBytes={try copyAI(last!,cfg.output_shape)}
                try save(["per_operator_mapping":"unavailable","preferred":"neuralEngine"],out.appendingPathComponent("plan.json"))
            } else {throw Failure.invalid("runtime")}
            try save(["load_start_ns":loadStart,"load_end_ns":mono(),"runtime":cfg.runtime,"model_path":cfg.model_path],out.appendingPathComponent("runtime.json"))
            let names=["original","zero","negative","repeat","benchmark","benchmark_repeat"]
            var controls=[Call]()
            for i in 0..<inputs.count {
                let(a,b)=try await predict(i);let d=try lastBytes();let file="\(names[i]).raw"
                try d.write(to:out.appendingPathComponent(file));controls.append(Call(phase:names[i],start_ns:a,end_ns:b,output:file,sha256:sha(d)))
            }
            try JSONEncoder().encode(controls).write(to:out.appendingPathComponent("controls.json"),options:.atomic)
            if cfg.seconds>0 {
                while !FileManager.default.fileExists(atPath:out.appendingPathComponent("bench-go").path) {
                    if FileManager.default.fileExists(atPath:out.appendingPathComponent("stop").path) {try save(["closed":true,"measured":false],out.appendingPathComponent("CLOSED.json"));return}
                    try await Task.sleep(for:.milliseconds(20))
                }
                let w=mono();var warm=0
                while warm<10 || Double(mono()-w)/1e9<cfg.warm_seconds {_=try await predict(4);warm+=1}
                try await Task.sleep(for:.seconds(cfg.quiet_seconds))
                var clocks=[UInt64]();clocks.reserveCapacity(500000);let start=mono();var progress=start
                repeat {
                    let(a,b)=try await predict(4);clocks.append(a);clocks.append(b)
                    if mono()-progress>2_000_000_000 {try save(["calls":clocks.count/2,"monotonic_ns":mono()],out.appendingPathComponent("progress.json"));progress=mono()}
                } while Double(mono()-start)/1e9<cfg.seconds
                let end=mono();let final=try lastBytes();try final.write(to:out.appendingPathComponent("output_final.raw"))
                try clocks.withUnsafeBytes{Data($0)}.write(to:out.appendingPathComponent("clocks.u64le"))
                try save(["start_ns":start,"end_ns":end,"calls":clocks.count/2,"warmup_calls":warm,"clock_format":"little-endian uint64 pairs: API start,end","output_sha256":sha(final)],out.appendingPathComponent("BLOCK.json"))
            }
            try save(["closed":true,"measured":cfg.seconds>0],out.appendingPathComponent("CLOSED.json"))
        } catch {
            if let out=destination {try? save(["error":String(describing:error)],out.appendingPathComponent("error.json"))}
            fputs("\(error)\n",stderr);exit(1)
        }
    }
}
