module {
  coreai.graph @qdq_mul_scale_s16(%arg0: tensor<1x32x1x64xf16> {coreai.name = "x"}) -> (tensor<1x16x1x64xf16> {coreai.name = "dequantize_1"}) {
    %0 = coreai.constant dense<0.000000e+00> : tensor<f16>
    %1 = coreai.constant dense<1> : tensor<si32>
    %2 = coreai.constant dense<2147483647> : tensor<4xsi32>
    %3 = coreai.constant dense<[0, 16, 0, 0]> : tensor<4xsi32>
    %4 = coreai.constant dense<6.250000e-02> : tensor<f16>
    %5 = coreai.constant dense<0> : tensor<si8>
    %6 = coreai.constant dense<0> : tensor<4xsi32>
    %7 = coreai.constant dense<[2147483647, 16, 2147483647, 2147483647]> : tensor<4xsi32>
    %8 = coreai.constant dense<1> : tensor<4xsi32>
    %9 = coreai.slice %arg0, %6, %7, %8 : (tensor<1x32x1x64xf16>, tensor<4xsi32>, tensor<4xsi32>, tensor<4xsi32>) -> tensor<1x16x1x64xf16>
    %10 = coreai.slice %arg0, %3, %2, %8 : (tensor<1x32x1x64xf16>, tensor<4xsi32>, tensor<4xsi32>, tensor<4xsi32>) -> tensor<1x16x1x64xf16>
    %11 = coreai.quantize %10, %4, %5, %0, %1 : (tensor<1x16x1x64xf16>, tensor<f16>, tensor<si8>, tensor<f16>, tensor<si32>) -> tensor<1x16x1x64xsi8>
    %12 = coreai.dequantize %11, %4, %5, %0, %1 : (tensor<1x16x1x64xsi8>, tensor<f16>, tensor<si8>, tensor<f16>, tensor<si32>) -> tensor<1x16x1x64xf16>
    %13 = coreai.decomposable.broadcasting_mul %9, %12 : (tensor<1x16x1x64xf16>, tensor<1x16x1x64xf16>) -> tensor<1x16x1x64xf16>
    %14 = coreai.quantize %13, %4, %5, %0, %1 : (tensor<1x16x1x64xf16>, tensor<f16>, tensor<si8>, tensor<f16>, tensor<si32>) -> tensor<1x16x1x64xsi8>
    %15 = coreai.dequantize %14, %4, %5, %0, %1 : (tensor<1x16x1x64xsi8>, tensor<f16>, tensor<si8>, tensor<f16>, tensor<si32>) -> tensor<1x16x1x64xf16>
    coreai.output %15 : tensor<1x16x1x64xf16>
  }
}
