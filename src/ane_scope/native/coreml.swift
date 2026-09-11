// Adapted from workspace scripts/g1gc_coreml.swift; see docs/PROVENANCE.md.
import Foundation
import CoreML
import CryptoKit
import Darwin

struct Config: Codable {
    let model_path: String
    let input_path: String
    let shape: [Int]
    let output_shape: [Int]
    let input_name: String
    let output_name: String
    let warmup: Int
    let repeats: Int
    let controls_only: Bool?
    let expected_output_path: String?
}

struct CallRecord: Codable {
    let phase: String
    let index: Int
    let start_epoch_ns: UInt64
    let start_monotonic_ns: UInt64
    let end_epoch_ns: UInt64
    let end_monotonic_ns: UInt64
    let duration_ns: UInt64
    let output_file: String
    let output_sha256: String
    let output_nonzero_count: Int
    let output_count: Int
}

struct Snapshot: Codable {
    let point: String
    let physical_footprint: UInt64?
}
struct ControlsFile: Codable { let records: [CallRecord]; let snapshots: [Snapshot]; let output_strides: [Int] }
struct ResultFile: Codable {
    let calls: Int; let records: [CallRecord]; let snapshots: [Snapshot]
    let input_strides: [Int]; let output_strides: [Int]
    let compiler_url: String; let compute_units: String
    let compile_start_epoch_ns: UInt64; let compile_end_epoch_ns: UInt64
    let compile_start_monotonic_ns: UInt64; let compile_end_monotonic_ns: UInt64
    let load_start_epoch_ns: UInt64; let load_end_epoch_ns: UInt64
    let load_start_monotonic_ns: UInt64; let load_end_monotonic_ns: UInt64
}
struct PlanOperation: Codable {
    let function: String; let path: String; let operator_name: String
    let output_names: [String]; let preferred: String?; let supported: [String]?; let estimated_weight: Double?
}

@main
struct Host {
    static func nowMono() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
    static func nowEpoch() -> UInt64 { UInt64(Date().timeIntervalSince1970 * 1_000_000_000) }

    static func writeJSON<T: Encodable>(_ value: T, _ url: URL) throws {
        let data = try JSONEncoder().encode(value)
        try data.write(to: url, options: .atomic)
    }

    static func footprint() -> UInt64? {
        var info = task_vm_info_data_t(); var count = mach_msg_type_number_t(MemoryLayout.size(ofValue: info) / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) { p in
            p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return kr == KERN_SUCCESS ? info.phys_footprint : nil
    }

    static func sha(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }

    static func makeArray(_ bytes: Data, shape: [Int]) throws -> MLMultiArray {
        let count = shape.reduce(1, *)
        guard bytes.count == count * 2 else { throw NSError(domain: "Host", code: 2, userInfo: [NSLocalizedDescriptionKey: "input byte count does not match shape"]) }
        let ptr = UnsafeMutableRawPointer.allocate(byteCount: bytes.count, alignment: MemoryLayout<Float16>.alignment)
        bytes.withUnsafeBytes { ptr.copyMemory(from: $0.baseAddress!, byteCount: bytes.count) }
        let nsShape = shape.map(NSNumber.init)
        var strides = [Int](repeating: 1, count: shape.count)
        if shape.count > 1 { for i in stride(from: shape.count - 2, through: 0, by: -1) { strides[i] = strides[i + 1] * shape[i + 1] } }
        do {
            return try MLMultiArray(dataPointer: ptr, shape: nsShape, dataType: .float16, strides: strides.map(NSNumber.init), deallocator: { p in p.deallocate() })
        } catch { ptr.deallocate(); throw error }
    }

    static func copyOutput(_ output: MLMultiArray, expectedShape: [Int]) throws -> Data {
        guard output.dataType == .float16 else { throw NSError(domain: "Host", code: 3, userInfo: [NSLocalizedDescriptionKey: "output is not float16"]) }
        let shape = output.shape.map { $0.intValue }
        guard shape == expectedShape else { throw NSError(domain: "Host", code: 4, userInfo: [NSLocalizedDescriptionKey: "output shape \(shape) != expected output shape \(expectedShape)"]) }
        var contiguous = [Int](repeating: 1, count: shape.count)
        if shape.count > 1 { for i in stride(from: shape.count - 2, through: 0, by: -1) { contiguous[i] = contiguous[i + 1] * shape[i + 1] } }
        let actualStrides = output.strides.map { $0.intValue }
        let n = output.count
        var values = [Float16](repeating: 0, count: n)
        let ptr = output.dataPointer.assumingMemoryBound(to: Float16.self)
        for index in 0..<n {
            var remainder = index; var offset = 0
            for axis in 0..<shape.count { let coordinate = remainder / contiguous[axis]; remainder %= contiguous[axis]; offset += coordinate * actualStrides[axis] }
            values[index] = ptr[offset]
        }
        let data = values.withUnsafeBytes { Data($0) }
        try data.withUnsafeBytes { raw in
            let p = raw.bindMemory(to: UInt16.self)
            for i in 0..<n { if !Float16(bitPattern: p[i]).isFinite { throw NSError(domain: "Host", code: 6, userInfo: [NSLocalizedDescriptionKey: "non-finite output at \(i)"]) } }
        }
        return data
    }

    static func inspectPlan(_ plan: MLComputePlan, _ url: URL) throws {
        var rows = [PlanOperation]()
        func walk(_ block: MLModelStructure.Program.Block, function: String, path: String) {
            for (i, op) in block.operations.enumerated() {
                let usage = plan.deviceUsage(for: op)
                rows.append(PlanOperation(function: function, path: "\(path)/op\(i)", operator_name: op.operatorName, output_names: op.outputs.map(\.name), preferred: usage?.preferred.description, supported: usage?.supported.map(\.description), estimated_weight: plan.estimatedCost(of: op)?.weight))
                for (j, nested) in op.blocks.enumerated() { walk(nested, function: function, path: "\(path)/op\(i)/block\(j)") }
            }
        }
        switch plan.modelStructure {
        case .program(let program):
            for (name, function) in program.functions { walk(function.block, function: name, path: name) }
        default: break
        }
        try JSONEncoder().encode(rows).write(to: url, options: .atomic)
    }

    static func main() async {
        do {
            let args = CommandLine.arguments
            guard args.count == 3 else { throw NSError(domain: "Host", code: 1, userInfo: [NSLocalizedDescriptionKey: "usage: host config.json outputdir"]) }
            let configURL = URL(fileURLWithPath: args[1]); let out = URL(fileURLWithPath: args[2], isDirectory: true)
            try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
            let config = try JSONDecoder().decode(Config.self, from: Data(contentsOf: configURL))
            guard config.shape.count == 4, config.output_shape.count == 4, config.shape.allSatisfy({ $0 > 0 && $0 <= 16384 }), config.output_shape.allSatisfy({ $0 > 0 && $0 <= 16384 }), config.shape.reduce(1, *) <= 4194304, config.output_shape.reduce(1, *) <= 4194304, config.warmup >= 0, config.repeats >= 0 else { throw NSError(domain: "Host", code: 10, userInfo: [NSLocalizedDescriptionKey: "invalid bounded rank-4 host configuration"]) }
            let input = try Data(contentsOf: URL(fileURLWithPath: config.input_path))
            let tensor = try makeArray(input, shape: config.shape)
            let readyEpoch = nowEpoch(); let readyMono = nowMono()
            try JSONSerialization.data(withJSONObject: ["pid": ProcessInfo.processInfo.processIdentifier, "ready": true, "stage": "precompile", "epoch_ns": readyEpoch, "monotonic_ns": readyMono]).write(to: out.appendingPathComponent("ready.json"), options: .atomic)
            let go = out.appendingPathComponent("go")
            while !FileManager.default.fileExists(atPath: go.path) { usleep(10_000) }
            try? FileManager.default.removeItem(at: go)
            let compileStart = nowMono(); let compileEpoch = nowEpoch()
            let temporaryCompiled = try await MLModel.compileModel(at: URL(fileURLWithPath: config.model_path))
            let compilerURL = out.appendingPathComponent("compiled.mlmodelc", isDirectory: true)
            try FileManager.default.copyItem(at: temporaryCompiled, to: compilerURL)
            let compileEnd = nowMono(); let compileEndEpoch = nowEpoch()
            let loadStart = nowMono(); let loadEpoch = nowEpoch()
            let mlc = MLModelConfiguration(); mlc.computeUnits = .cpuAndNeuralEngine
            let model = try MLModel(contentsOf: compilerURL, configuration: mlc)
            let description = model.modelDescription
            guard description.inputDescriptionsByName.count == 1, description.outputDescriptionsByName.count == 1,
                  let inputDescription = description.inputDescriptionsByName[config.input_name]?.multiArrayConstraint,
                  let outputDescription = description.outputDescriptionsByName[config.output_name]?.multiArrayConstraint,
                  inputDescription.shape.map({ $0.intValue }) == config.shape,
                  outputDescription.shape.map({ $0.intValue }) == config.output_shape,
                  inputDescription.dataType == .float16, outputDescription.dataType == .float16,
                  model.configuration.computeUnits == .cpuAndNeuralEngine else {
                throw NSError(domain: "Host", code: 11, userInfo: [NSLocalizedDescriptionKey: "actual Core ML descriptor/configuration differs from registered asset"])
            }
            try JSONSerialization.data(withJSONObject: ["input_name":config.input_name,"output_name":config.output_name,"input_shape":inputDescription.shape,"output_shape":outputDescription.shape,"input_type":"float16","output_type":"float16","actual_compute_units":"cpuAndNeuralEngine","loaded_model_path":compilerURL.path,"source_package_path":config.model_path]).write(to:out.appendingPathComponent("runtime.json"),options:.atomic)
            let loadEnd = nowMono(); let loadEndEpoch = nowEpoch()
            do {
                let plan = try await MLComputePlan.load(contentsOf: compilerURL, configuration: mlc)
                try inspectPlan(plan, out.appendingPathComponent("plan.json"))
            } catch {
                try? JSONSerialization.data(withJSONObject: ["error": "compute plan inspection failed: \(error)"]).write(to: out.appendingPathComponent("plan.json"), options: .atomic)
                throw error
            }
            var snapshots = [Snapshot(point: "loaded", physical_footprint: footprint())]
            var records = [CallRecord](); var totalCalls = 0
            var outputStrides = [Int]()
            func persistCalls() throws { try JSONEncoder().encode(records).write(to: out.appendingPathComponent("calls.json"), options: .atomic) }
            func call(_ phase: String, _ index: Int, _ array: MLMultiArray, retainFile: String?) throws {
                let result: (Data, UInt64, UInt64, UInt64, UInt64) = try autoreleasepool {
                    let epoch = nowEpoch(); let mono = nowMono(); let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: [config.input_name: MLFeatureValue(multiArray: array)]))
                    let end = nowMono(); let endEpoch = nowEpoch(); guard let ma = output.featureValue(for: config.output_name)?.multiArrayValue else { throw NSError(domain: "Host", code: 5, userInfo: [NSLocalizedDescriptionKey: "missing output \(config.output_name)"]) }
                    outputStrides = ma.strides.map(\.intValue)
                    let raw = try copyOutput(ma, expectedShape: config.output_shape); return (raw, epoch, mono, endEpoch, end)
                }
                if let file = retainFile { try result.0.write(to: out.appendingPathComponent(file), options: .atomic) }
                let nonzero = result.0.withUnsafeBytes { $0.bindMemory(to: UInt16.self).reduce(0) { $0 + (($1 & 0x7fff) != 0 ? 1 : 0) } }
                records.append(CallRecord(phase: phase, index: index, start_epoch_ns: result.1, start_monotonic_ns: result.2, end_epoch_ns: result.3, end_monotonic_ns: result.4, duration_ns: result.4 - result.2, output_file: retainFile ?? "", output_sha256: sha(result.0), output_nonzero_count: nonzero, output_count: result.0.count / 2)); totalCalls += 1
                try persistCalls()
            }
            try call("control_original", 0, try makeArray(input, shape: config.shape), retainFile: "control_original.raw")
            let zero = try makeArray(Data(repeating: 0, count: input.count), shape: config.shape)
            try call("control_zero", 0, zero, retainFile: "control_zero.raw")
            var negative = input
            negative.withUnsafeMutableBytes { raw in
                let p = raw.bindMemory(to: UInt16.self); for i in 0..<p.count { p[i] ^= 0x8000 }
            }
            try call("control_negative", 0, try makeArray(negative, shape: config.shape), retainFile: "control_negative.raw")
            try call("control_original_repeat", 0, try makeArray(input, shape: config.shape), retainFile: "control_original_repeat.raw")
            snapshots.append(Snapshot(point: "controls", physical_footprint: footprint()))
            try JSONEncoder().encode(ControlsFile(records: records, snapshots: snapshots, output_strides: outputStrides)).write(to: out.appendingPathComponent("controls.json"), options: .atomic)
            if !(config.controls_only ?? false) {
                let benchGo = out.appendingPathComponent("bench-go")
                while !FileManager.default.fileExists(atPath: benchGo.path) { usleep(10_000) }
                try? FileManager.default.removeItem(at: benchGo)
                for i in 0..<config.warmup { try call("warmup", i, tensor, retainFile: nil) }
                snapshots.append(Snapshot(point: "warm", physical_footprint: footprint()))
                for i in 0..<config.repeats { try call("measured", i, tensor, retainFile: i == 0 ? "output_first.raw" : (i + 1 == config.repeats ? "output_final.raw" : nil)) }
                snapshots.append(Snapshot(point: "measured", physical_footprint: footprint()))
            }
            let result = ResultFile(calls: totalCalls, records: records, snapshots: snapshots, input_strides: tensor.strides.map(\.intValue), output_strides: outputStrides, compiler_url: compilerURL.path, compute_units: "cpuAndNeuralEngine", compile_start_epoch_ns: compileEpoch, compile_end_epoch_ns: compileEndEpoch, compile_start_monotonic_ns: compileStart, compile_end_monotonic_ns: compileEnd, load_start_epoch_ns: loadEpoch, load_end_epoch_ns: loadEndEpoch, load_start_monotonic_ns: loadStart, load_end_monotonic_ns: loadEnd)
            try JSONEncoder().encode(result).write(to: out.appendingPathComponent("result.json"), options: .atomic)
        } catch {
            let out = CommandLine.arguments.count > 2 ? URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true) : nil
            if let out { try? FileManager.default.createDirectory(at: out, withIntermediateDirectories: true); try? JSONSerialization.data(withJSONObject: ["error": String(describing: error)]).write(to: out.appendingPathComponent("error.json"), options: .atomic) }
            fputs("ERROR: \(error)\n", stderr); exit(1)
        }
    }
}
