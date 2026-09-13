module {
  coreai.graph @qdq_mul_scale_s4_clip(%arg0: tensor<1x32x1x64xf16> {coreai.name = "x"}) -> (tensor<1x16x1x64xf16> {coreai.name = "dequantize_1"}) {
    %0 = coreai.constant dense<3.175000e+01> : tensor<f16>
    %1 = coreai.constant dense<-3.200000e+01> : tensor<f16>
    %2 = coreai.constant dense<0.000000e+00> : tensor<f16>
    %3 = coreai.constant dense<1> : tensor<si32>
    %4 = coreai.constant dense<2147483647> : tensor<4xsi32>
    %5 = coreai.constant dense<[0, 16, 0, 0]> : tensor<4xsi32>
    %6 = coreai.constant dense<6.250000e-02> : tensor<f16>
    %7 = coreai.constant dense<2.500000e-01> : tensor<f16>
    %8 = coreai.constant dense<0> : tensor<si8>
    %9 = coreai.constant dense<0> : tensor<4xsi32>
    %10 = coreai.constant dense<[2147483647, 16, 2147483647, 2147483647]> : tensor<4xsi32>
    %11 = coreai.constant dense<1> : tensor<4xsi32>
    %12 = coreai.slice %arg0, %9, %10, %11 : (tensor<1x32x1x64xf16>, tensor<4xsi32>, tensor<4xsi32>, tensor<4xsi32>) -> tensor<1x16x1x64xf16>
    %13 = coreai.slice %arg0, %5, %4, %11 : (tensor<1x32x1x64xf16>, tensor<4xsi32>, tensor<4xsi32>, tensor<4xsi32>) -> tensor<1x16x1x64xf16>
    %14 = coreai.quantize %13, %6, %8, %2, %3 : (tensor<1x16x1x64xf16>, tensor<f16>, tensor<si8>, tensor<f16>, tensor<si32>) -> tensor<1x16x1x64xsi8>
    %15 = coreai.dequantize %14, %6, %8, %2, %3 : (tensor<1x16x1x64xsi8>, tensor<f16>, tensor<si8>, tensor<f16>, tensor<si32>) -> tensor<1x16x1x64xf16>
    %16 = coreai.decomposable.broadcasting_mul %12, %15 : (tensor<1x16x1x64xf16>, tensor<1x16x1x64xf16>) -> tensor<1x16x1x64xf16>
    %17 = coreai.decomposable.broadcasting_maximum %16, %1 : (tensor<1x16x1x64xf16>, tensor<f16>) -> tensor<1x16x1x64xf16>
    %18 = coreai.decomposable.broadcasting_minimum %17, %0 : (tensor<1x16x1x64xf16>, tensor<f16>) -> tensor<1x16x1x64xf16>
    %19 = coreai.quantize %18, %7, %8, %2, %3 : (tensor<1x16x1x64xf16>, tensor<f16>, tensor<si8>, tensor<f16>, tensor<si32>) -> tensor<1x16x1x64xsi8>
    %20 = coreai.dequantize %19, %7, %8, %2, %3 : (tensor<1x16x1x64xsi8>, tensor<f16>, tensor<si8>, tensor<f16>, tensor<si32>) -> tensor<1x16x1x64xf16>
    coreai.output %20 : tensor<1x16x1x64xf16>
  }
}
