module {
  coreai.graph @qdq_mul_scale_s8(%arg0: tensor<1x32x1x64xf16> {coreai.name = "x"}) -> (tensor<1x16x1x64xf16> {coreai.name = "dequantize_1"}) {
    %0 = coreai.constant dense<0.000000e+00> : tensor<f16>
    %1 = coreai.constant dense<1> : tensor<si32>
    %2 = coreai.constant dense<2147483647> : tensor<4xsi32>
    %3 = coreai.constant dense<[0, 16, 0, 0]> : tensor<4xsi32>
    %4 = coreai.constant dense<6.250000e-02> : tensor<f16>
    %5 = coreai.constant dense<1.250000e-01> : tensor<f16>
    %6 = coreai.constant dense<0> : tensor<si8>
    %7 = coreai.constant dense<0> : tensor<4xsi32>
    %8 = coreai.constant dense<[2147483647, 16, 2147483647, 2147483647]> : tensor<4xsi32>
    %9 = coreai.constant dense<1> : tensor<4xsi32>
    %10 = coreai.slice %arg0, %7, %8, %9 : (tensor<1x32x1x64xf16>, tensor<4xsi32>, tensor<4xsi32>, tensor<4xsi32>) -> tensor<1x16x1x64xf16>
    %11 = coreai.slice %arg0, %3, %2, %9 : (tensor<1x32x1x64xf16>, tensor<4xsi32>, tensor<4xsi32>, tensor<4xsi32>) -> tensor<1x16x1x64xf16>
    %12 = coreai.quantize %11, %4, %6, %0, %1 : (tensor<1x16x1x64xf16>, tensor<f16>, tensor<si8>, tensor<f16>, tensor<si32>) -> tensor<1x16x1x64xsi8>
    %13 = coreai.dequantize %12, %4, %6, %0, %1 : (tensor<1x16x1x64xsi8>, tensor<f16>, tensor<si8>, tensor<f16>, tensor<si32>) -> tensor<1x16x1x64xf16>
    %14 = coreai.decomposable.broadcasting_mul %10, %13 : (tensor<1x16x1x64xf16>, tensor<1x16x1x64xf16>) -> tensor<1x16x1x64xf16>
    %15 = coreai.quantize %14, %5, %6, %0, %1 : (tensor<1x16x1x64xf16>, tensor<f16>, tensor<si8>, tensor<f16>, tensor<si32>) -> tensor<1x16x1x64xsi8>
    %16 = coreai.dequantize %15, %5, %6, %0, %1 : (tensor<1x16x1x64xsi8>, tensor<f16>, tensor<si8>, tensor<f16>, tensor<si32>) -> tensor<1x16x1x64xf16>
    coreai.output %16 : tensor<1x16x1x64xf16>
  }
}
