import Foundation
import CoreAI
import CryptoKit
import Darwin

struct Config: Decodable {
    let assets: [String:String]
    let entries: [String:String]
    let input: String
    let expected: [String:String]
    let pairs: Int
    let warmup: Int
    let repeats: Int
    let order: [String]
    let probe: Bool
    let inputChannels: Int
    let outputChannels: Int
}
struct Record: Codable {
    let arm: String
    let phase: String
    let pair: Int
    let index: Int
    let zero: Bool
    let start_epoch: Double
    let run_start_epoch: Double
    let run_end_epoch: Double
    let run_ns: UInt64
    let total_ns: UInt64
    let output_exact: Bool
}
func writeJSON<T: Encodable>(_ data: T, _ url: URL) throws {
    let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted,.sortedKeys]
    try encoder.encode(data).write(to:url,options:.atomic)
}
func footprint() -> UInt64 {
    var info=task_vm_info_data_t(), count=mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size/MemoryLayout<integer_t>.size)
    let result=withUnsafeMutablePointer(to:&info) { p in p.withMemoryRebound(to:integer_t.self,capacity:Int(count)) { task_info(mach_task_self_,task_flavor_t(TASK_VM_INFO),$0,&count) } }
    precondition(result == KERN_SUCCESS); return info.phys_footprint
}
final class MLP {
    var model: AIModel?
    var fn: InferenceFunction?
    let inputChannels: Int, outputChannels: Int
    var input: NDArray
    var copied: [Float16]
    let expected: Data
    let probe: Bool
    let zeroExpected: Data
    var inputName = "", outputName = ""
    init(expected:Data, probe:Bool, inputChannels:Int, outputChannels:Int) {
        self.expected=expected; self.probe=probe; self.inputChannels=inputChannels; self.outputChannels=outputChannels
        input=NDArray(shape:[1,inputChannels,1,64],scalarType:.float16,strides:[inputChannels*64,64,64,1])
        copied=[Float16](repeating:0,count:outputChannels*64)
        zeroExpected=Data(count:outputChannels*64*2)
        precondition(expected.count==outputChannels*64*2)
    }
    func load(_ path:String, _ entry:String) async throws {
        model = try await AIModel(contentsOf:URL(fileURLWithPath:path),options:SpecializationOptions(preferredComputeUnitKind:.neuralEngine))
        precondition(model!.functionNames == [entry]); fn = try model!.loadFunction(named:entry)!
        let d=fn!.descriptor; precondition(d.inputNames.count==1 && d.outputNames.count==1 && d.stateNames.isEmpty)
        inputName=d.inputNames[0]; outputName=d.outputNames[0]
    }
    func call(_ data:Data, arm:String, phase:String, pair:Int, index:Int, zero:Bool, retain:URL?) async throws -> Record {
        let se=Date().timeIntervalSince1970, begin=DispatchTime.now().uptimeNanoseconds
        input.mutableView(as:Float16.self).withUnsafeMutablePointer { p,shape,strides in
            precondition(shape[1]==inputChannels && shape[3]==64 && strides[1]==64 && strides[3]==1)
            _ = data.withUnsafeBytes { raw in memcpy(p,raw.baseAddress!,data.count) }
        }
        let rs=Date().timeIntervalSince1970, tick=DispatchTime.now().uptimeNanoseconds
        var outputs=try await fn!.run(inputs:[inputName:input])
        let runEnd=DispatchTime.now().uptimeNanoseconds, re=Date().timeIntervalSince1970
        try autoreleasepool {
            guard let array=outputs.remove(outputName)?.ndArray else { throw NSError(domain:"missing_output",code:1) }
            precondition(array.scalarType == .float16)
            array.view(as:Float16.self).withUnsafePointer { p,shape,strides in
                precondition(shape.count==4 && shape[0]==1 && shape[1]==outputChannels && shape[2]==1 && shape[3]==64)
                copied.withUnsafeMutableBufferPointer { dst in
                    if strides[1]==64 && strides[3]==1 { dst.baseAddress!.update(from:p,count:dst.count) }
                    else { for c in 0..<outputChannels { for t in 0..<64 { dst[c*64+t]=p[c*strides[1]+t*strides[3]] } } }
                }
            }
        }
        let end=DispatchTime.now().uptimeNanoseconds
        let comparison=zero ? zeroExpected : expected
        let exact=copied.withUnsafeBytes { actual in comparison.withUnsafeBytes { ref in memcmp(actual.baseAddress!,ref.baseAddress!,ref.count)==0 } }
        if let file=retain { try copied.withUnsafeBytes { try Data($0).write(to:file,options:.atomic) } }
        guard exact || probe else { throw NSError(domain:"output_bytes_mismatch_\(arm)_\(phase)",code:2) }
        return Record(arm:arm,phase:phase,pair:pair,index:index,zero:zero,start_epoch:se,run_start_epoch:rs,run_end_epoch:re,run_ns:runEnd-tick,total_ns:end-begin,output_exact:exact)
    }
    func release() { fn=nil; model=nil }
}
@main struct Host {
    static func main() async throws {
        let args=CommandLine.arguments; precondition(args.count==3)
        let cfg=try JSONDecoder().decode(Config.self,from:Data(contentsOf:URL(fileURLWithPath:args[1])))
        let out=URL(fileURLWithPath:args[2]); try FileManager.default.createDirectory(at:out,withIntermediateDirectories:true)
        try writeJSON(["pid":Int(getpid())],out.appendingPathComponent("ready.json"))
        while !FileManager.default.fileExists(atPath:out.appendingPathComponent("go").path) { try await Task.sleep(for:.milliseconds(10)) }
        let original=try Data(contentsOf:URL(fileURLWithPath:cfg.input)); precondition(original.count==cfg.inputChannels*64*2)
        let zero=Data(count:original.count)
        var gates=[String:MLP](), records=[Record](), snapshots=[[String:UInt64]]()
        let order=cfg.order
        precondition(Set(order)==Set(cfg.assets.keys) && order.count==cfg.assets.count)
        var loads=[String:Double]()
        for arm in order {
            let gate=MLP(expected:try Data(contentsOf:URL(fileURLWithPath:cfg.expected[arm]!)),probe:cfg.probe,inputChannels:cfg.inputChannels,outputChannels:cfg.outputChannels)
            let start=Date().timeIntervalSince1970; try await gate.load(cfg.assets[arm]!,cfg.entries[arm]!)
            loads[arm]=Date().timeIntervalSince1970-start; gates[arm]=gate
        }
        try writeJSON(loads,out.appendingPathComponent("loads.json"))
        snapshots.append(["loaded":footprint()])
        for arm in order {
            for (i,isZero) in [false,true,false].enumerated() {
                records.append(try await gates[arm]!.call(isZero ? zero : original,arm:arm,phase:"control",pair:-1,index:i,zero:isZero,retain:out.appendingPathComponent("\(arm)-control-\(i).raw")))
            }
        }
        try writeJSON(records,out.appendingPathComponent("calls.json"))
        for pair in 0..<cfg.pairs {
            let arms=pair % 2 == 0 ? order : Array(order.reversed())
            for arm in arms {
                for i in 0..<cfg.warmup { records.append(try await gates[arm]!.call(original,arm:arm,phase:"warmup",pair:pair,index:i,zero:false,retain:nil)) }
                for i in 0..<cfg.repeats { records.append(try await gates[arm]!.call(original,arm:arm,phase:"measured",pair:pair,index:i,zero:false,retain:nil)) }
                try writeJSON(records,out.appendingPathComponent("calls.json"))
                snapshots.append(["pair_\(pair)_\(arm)":footprint()])
            }
        }
        for gate in gates.values { gate.release() }; gates.removeAll()
        snapshots.append(["released":footprint()])
        try writeJSON(snapshots,out.appendingPathComponent("footprints.json"))
        try JSONSerialization.data(withJSONObject:["pid":getpid(),"calls":records.count,"contexts_closed":true,"loads":cfg.assets.count,"all_outputs_exact":records.allSatisfy { $0.output_exact },"thermal_state":ProcessInfo.processInfo.thermalState.rawValue],options:[.prettyPrinted,.sortedKeys]).write(to:out.appendingPathComponent("stats.json"),options:.atomic)
    }
}
