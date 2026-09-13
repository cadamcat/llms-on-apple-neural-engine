module {
  coreai.graph @qdq_mul_scale_s16_clip(%arg0: tensor<1x32x1x64xf16> {coreai.name = "x"}) -> (tensor<1x16x1x64xf16> {coreai.name = "dequantize_1"}) {
    %0 = coreai.constant dense<7.937500e+00> : tensor<f16>
    %1 = coreai.constant dense<-8.000000e+00> : tensor<f16>
    %2 = coreai.constant dense<0.000000e+00> : tensor<f16>
    %3 = coreai.constant dense<1> : tensor<si32>
    %4 = coreai.constant dense<2147483647> : tensor<4xsi32>
    %5 = coreai.constant dense<[0, 16, 0, 0]> : tensor<4xsi32>
    %6 = coreai.constant dense<6.250000e-02> : tensor<f16>
    %7 = coreai.constant dense<0> : tensor<si8>
    %8 = coreai.constant dense<0> : tensor<4xsi32>
    %9 = coreai.constant dense<[2147483647, 16, 2147483647, 2147483647]> : tensor<4xsi32>
    %10 = coreai.constant dense<1> : tensor<4xsi32>
    %11 = coreai.slice %arg0, %8, %9, %10 : (tensor<1x32x1x64xf16>, tensor<4xsi32>, tensor<4xsi32>, tensor<4xsi32>) -> tensor<1x16x1x64xf16>
    %12 = coreai.slice %arg0, %5, %4, %10 : (tensor<1x32x1x64xf16>, tensor<4xsi32>, tensor<4xsi32>, tensor<4xsi32>) -> tensor<1x16x1x64xf16>
    %13 = coreai.quantize %12, %6, %7, %2, %3 : (tensor<1x16x1x64xf16>, tensor<f16>, tensor<si8>, tensor<f16>, tensor<si32>) -> tensor<1x16x1x64xsi8>
    %14 = coreai.dequantize %13, %6, %7, %2, %3 : (tensor<1x16x1x64xsi8>, tensor<f16>, tensor<si8>, tensor<f16>, tensor<si32>) -> tensor<1x16x1x64xf16>
    %15 = coreai.decomposable.broadcasting_mul %11, %14 : (tensor<1x16x1x64xf16>, tensor<1x16x1x64xf16>) -> tensor<1x16x1x64xf16>
    %16 = coreai.decomposable.broadcasting_maximum %15, %1 : (tensor<1x16x1x64xf16>, tensor<f16>) -> tensor<1x16x1x64xf16>
    %17 = coreai.decomposable.broadcasting_minimum %16, %0 : (tensor<1x16x1x64xf16>, tensor<f16>) -> tensor<1x16x1x64xf16>
    %18 = coreai.quantize %17, %6, %7, %2, %3 : (tensor<1x16x1x64xf16>, tensor<f16>, tensor<si8>, tensor<f16>, tensor<si32>) -> tensor<1x16x1x64xsi8>
    %19 = coreai.dequantize %18, %6, %7, %2, %3 : (tensor<1x16x1x64xsi8>, tensor<f16>, tensor<si8>, tensor<f16>, tensor<si32>) -> tensor<1x16x1x64xf16>
    coreai.output %19 : tensor<1x16x1x64xf16>
  }
}
