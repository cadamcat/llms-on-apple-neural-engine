// ANE Scope Swift host. See docs/PROVENANCE.md for experiment origins.
import Foundation
import CoreAI
import CryptoKit
import Darwin

struct Config: Codable { let model_path: String; let input_path: String; let shape: [Int]; let output_shape: [Int]; let input_name: String?; let output_name: String?; let entrypoint: String; let warmup: Int; let repeats: Int; let controls_only: Bool? }
struct CallRecord: Codable { let phase: String; let index: Int; let start_epoch_ns: UInt64; let start_monotonic_ns: UInt64; let end_epoch_ns: UInt64; let end_monotonic_ns: UInt64; let duration_ns: UInt64; let output_file: String; let output_sha256: String; let output_nonzero_count: Int; let output_count: Int }
struct Snapshot: Codable { let point: String; let physical_footprint: UInt64? }
struct ControlsFile: Codable { let records: [CallRecord]; let snapshots: [Snapshot]; let output_strides: [Int] }
struct ResultFile: Codable { let calls: Int; let records: [CallRecord]; let snapshots: [Snapshot]; let input_strides: [Int]; let output_strides: [Int]; let function_names: [String]; let entrypoint: String; let input_name: String; let output_name: String; let preferred_compute_unit: String; let plan_status: String }

@main struct Host {
    static func mono() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
    static func epoch() -> UInt64 { UInt64(Date().timeIntervalSince1970 * 1_000_000_000) }
    static func sha(_ d: Data) -> String { SHA256.hash(data: d).map { String(format: "%02x", $0) }.joined() }
    static func footprint() -> UInt64? {
        var info = task_vm_info_data_t(); var count = mach_msg_type_number_t(MemoryLayout.size(ofValue: info) / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) { p in p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) } }
        return kr == KERN_SUCCESS ? info.phys_footprint : nil
    }
    static func readHalf(_ url: URL) throws -> [Float16] { let d = try Data(contentsOf: url); guard d.count % 2 == 0 else { throw NSError(domain: "Host", code: 2, userInfo: [NSLocalizedDescriptionKey: "input is not FP16 aligned"]) }; return d.withUnsafeBytes { raw in (0..<(d.count / 2)).map { Float16(bitPattern: raw.loadUnaligned(fromByteOffset: $0 * 2, as: UInt16.self)) } } }
    static func halfData(_ x: [Float16]) -> Data { x.withUnsafeBytes { Data($0) } }
    static func make(_ shape: [Int]) -> NDArray { let s = [shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1]; return NDArray(shape: shape, scalarType: .float16, strides: s) }
    static func fill(_ a: inout NDArray, _ values: [Float16]) { a.mutableView(as: Float16.self).withUnsafeMutablePointer { p, shape, strides in let dims = (0..<shape.count).map { shape[$0] }; precondition(dims.count == 4 && dims.reduce(1, *) == values.count); let expected = [dims[1]*dims[2]*dims[3],dims[2]*dims[3],dims[3],1]; precondition((0..<strides.count).map { strides[$0] } == expected); values.withUnsafeBufferPointer { p.update(from: $0.baseAddress!, count: values.count) } } }
    static func copy(_ a: NDArray, shape: [Int]) throws -> Data {
        guard a.scalarType == .float16 else { throw NSError(domain: "Host", code: 3, userInfo: [NSLocalizedDescriptionKey: "output is not FP16"]) }
        return try a.view(as: Float16.self).withUnsafePointer { p, actual, strides in
            guard (0..<actual.count).map({ actual[$0] }) == shape else { throw NSError(domain: "Host", code: 4, userInfo: [NSLocalizedDescriptionKey: "output shape mismatch"]) }
            var x = [Float16](repeating: 0, count: shape.reduce(1, *)); for n in 0..<x.count { let i0 = n / (shape[1]*shape[2]*shape[3]); let r = n % (shape[1]*shape[2]*shape[3]); let i1 = r / (shape[2]*shape[3]); let q = r % (shape[2]*shape[3]); let i2 = q / shape[3]; let i3 = q % shape[3]; x[n] = p[i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3]] }
            for (i, v) in x.enumerated() where !v.isFinite { throw NSError(domain: "Host", code: 5, userInfo: [NSLocalizedDescriptionKey: "non-finite output at \(i)"]) }
            return halfData(x)
        }
    }
    static func main() async {
        do {
            let a = CommandLine.arguments; guard a.count == 3 else { throw NSError(domain: "Host", code: 1, userInfo: [NSLocalizedDescriptionKey: "usage: host config.json outputdir"]) }
            let cfg = try JSONDecoder().decode(Config.self, from: Data(contentsOf: URL(fileURLWithPath: a[1]))); let out = URL(fileURLWithPath: a[2], isDirectory: true); try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
            let input = try readHalf(URL(fileURLWithPath: cfg.input_path)); guard cfg.shape.count == 4 && cfg.output_shape.count == 4 && cfg.shape.allSatisfy({ $0 > 0 && $0 <= 16384 }) && cfg.output_shape.allSatisfy({ $0 > 0 && $0 <= 16384 }) && cfg.shape.reduce(1, *) <= 4194304 && cfg.output_shape.reduce(1, *) <= 4194304 && input.count == cfg.shape.reduce(1, *) else { throw NSError(domain: "Host", code: 6, userInfo: [NSLocalizedDescriptionKey: "shape/input mismatch"]) }
            try JSONSerialization.data(withJSONObject: ["pid": getpid(), "ready": true, "epoch_ns": epoch(), "monotonic_ns": mono()]).write(to: out.appendingPathComponent("ready.json"), options: .atomic)
            while !FileManager.default.fileExists(atPath: out.appendingPathComponent("go").path) { try await Task.sleep(for: .milliseconds(10)) }; try? FileManager.default.removeItem(at: out.appendingPathComponent("go"))
            let loadBegin = mono(); let model = try await AIModel(contentsOf: URL(fileURLWithPath: cfg.model_path), options: SpecializationOptions(preferredComputeUnitKind: .neuralEngine)); guard model.functionNames == [cfg.entrypoint], let fn = try model.loadFunction(named: cfg.entrypoint) else { throw NSError(domain: "Host", code: 11, userInfo: [NSLocalizedDescriptionKey: "actual function does not match requested entrypoint"]) }
            let d = fn.descriptor; guard d.inputNames.count == 1 && d.outputNames.count == 1 && d.stateNames.isEmpty else { throw NSError(domain: "Host", code: 7, userInfo: [NSLocalizedDescriptionKey: "descriptor must have one input/output and no state"]) }
            let inputName = d.inputNames[0], outputName = d.outputNames[0]; if let n = cfg.input_name, n != inputName { throw NSError(domain: "Host", code: 9, userInfo: [NSLocalizedDescriptionKey: "input name mismatch"]) }; if let n = cfg.output_name, n != outputName { throw NSError(domain: "Host", code: 10, userInfo: [NSLocalizedDescriptionKey: "output name mismatch"]) }
            try JSONSerialization.data(withJSONObject: ["function_names":model.functionNames,"entrypoint":cfg.entrypoint,"input_name":inputName,"output_name":outputName,"preferred_compute_unit":"neuralEngine","specialization_load_ns":mono()-loadBegin,"per_operation_mapping":"unverified"]).write(to:out.appendingPathComponent("runtime.json"),options:.atomic)
            try JSONSerialization.data(withJSONObject:["status":"not_available_in_this_host","per_operation_mapping":"unverified"]).write(to:out.appendingPathComponent("plan.json"),options:.atomic)
            var orig = make(cfg.shape); fill(&orig, input); var zero = make(cfg.shape); fill(&zero, [Float16](repeating: 0, count: input.count)); var neg = make(cfg.shape); fill(&neg, input.map { Float16(bitPattern: $0.bitPattern ^ 0x8000) })
            var records = [CallRecord](), snapshots = [Snapshot(point: "loaded", physical_footprint: footprint())], outputStrides = [Int](), total = 0
            func persist() throws { try JSONEncoder().encode(records).write(to: out.appendingPathComponent("calls.json"), options: .atomic) }
            func call(_ phase: String, _ index: Int, _ x: NDArray, retain: String?) async throws {
                let se = epoch(), sm = mono(); var outputs = try await fn.run(inputs: [inputName: x]); let em = mono(), ee = epoch()
                let raw: Data = try autoreleasepool { guard let v = outputs.remove(outputName)?.ndArray else { throw NSError(domain: "Host", code: 8, userInfo: [NSLocalizedDescriptionKey: "missing output"]) }; outputStrides = v.view(as: Float16.self).withUnsafePointer { _, _, strides in (0..<strides.count).map { strides[$0] } }; return try copy(v, shape: cfg.output_shape) }
                if let r = retain { try raw.write(to: out.appendingPathComponent(r), options: .atomic) }; let nz = raw.withUnsafeBytes { $0.bindMemory(to: UInt16.self).reduce(0) { $0 + (($1 & 0x7fff) != 0 ? 1 : 0) } }; records.append(CallRecord(phase: phase, index: index, start_epoch_ns: se, start_monotonic_ns: sm, end_epoch_ns: ee, end_monotonic_ns: em, duration_ns: em-sm, output_file: retain ?? "", output_sha256: sha(raw), output_nonzero_count: nz, output_count: raw.count/2)); total += 1; try persist()
            }
            try await call("control_original", 0, orig, retain: "control_original.raw"); try await call("control_zero", 0, zero, retain: "control_zero.raw"); try await call("control_negative", 0, neg, retain: "control_negative.raw"); try await call("control_original_repeat", 0, orig, retain: "control_original_repeat.raw")
            snapshots.append(Snapshot(point: "controls", physical_footprint: footprint())); try JSONEncoder().encode(ControlsFile(records: records, snapshots: snapshots, output_strides: outputStrides)).write(to: out.appendingPathComponent("controls.json"), options: .atomic)
            if !(cfg.controls_only ?? false) { while !FileManager.default.fileExists(atPath: out.appendingPathComponent("bench-go").path) { try await Task.sleep(for: .milliseconds(10)) }; try? FileManager.default.removeItem(at: out.appendingPathComponent("bench-go")); for i in 0..<cfg.warmup { try await call("warmup", i, orig, retain: nil) }; snapshots.append(Snapshot(point: "warm", physical_footprint: footprint())); for i in 0..<cfg.repeats { try await call("measured", i, orig, retain: i == 0 ? "output_first.raw" : (i + 1 == cfg.repeats ? "output_final.raw" : nil)) }; snapshots.append(Snapshot(point: "measured", physical_footprint: footprint()) ) }
            let result = ResultFile(calls: total, records: records, snapshots: snapshots, input_strides: [cfg.shape[1]*cfg.shape[2]*cfg.shape[3],cfg.shape[2]*cfg.shape[3],cfg.shape[3],1], output_strides: outputStrides, function_names: model.functionNames, entrypoint: cfg.entrypoint, input_name: inputName, output_name: outputName, preferred_compute_unit: "neuralEngine", plan_status: "not_available_publicly")
            try JSONEncoder().encode(result).write(to: out.appendingPathComponent("result.json"), options: .atomic); try JSONSerialization.data(withJSONObject: ["status": "not_available_publicly"]).write(to: out.appendingPathComponent("plan.json"), options: .atomic)
        } catch { if CommandLine.arguments.count > 2 { let o = URL(fileURLWithPath: CommandLine.arguments[2]); try? FileManager.default.createDirectory(at: o, withIntermediateDirectories: true); try? JSONSerialization.data(withJSONObject: ["error": String(describing: error)]).write(to: o.appendingPathComponent("error.json"), options: .atomic) }; fputs("ERROR: \(error)\n", stderr); exit(1) }
    }
}
