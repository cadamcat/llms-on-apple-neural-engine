module {
  coreai.graph @scope_coreai_group_native64(%arg0: tensor<1x64x1x64xf16> {coreai.name = "x"}) -> (tensor<1x64x1x64xf16> {coreai.name = "convolution"}) {
    %0 = coreai.constant dense<1> : tensor<ui32>
    %1 = coreai.constant dense<1> : tensor<2xui32>
    %2 = coreai.constant dense<0.000000e+00> : tensor<64x2x1x1xf16>
    %3 = coreai.constant dense<0> : tensor<si16>
    %4 = coreai.constant dense<0.000000e+00> : tensor<64xf16>
    %5 = coreai.constant dense<[2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01, 5.000000e-01]> : tensor<64xf16>
    %6 = coreai.constant dense<0> : tensor<64xsi8>
    %7 = coreai.constant dense_resource<resource_1637340467108146368> : tensor<64x64x1x1xui4>
    %8 = coreai.constant dense<[[[[[[-8], [-7], [-6], [-5], [-4], [-3], [-2], [-1], [0], [1], [2], [3], [4], [5], [6], [7]]]]]]> : tensor<1x1x1x1x16x1xsi8>
    %9 = coreai.constant dense<"0x00300034003400300030003400340030003000340034003000300034003400300030003400340030003000340034003000300034003400300030003400340030003000340034003000300034003400300030003400340030003000340034003000300034003400300030003400340030003000340034003000300034003400300030003400340030003000340034003000300034003400300030003400340030003000340034003000300034003400300030003400340030003000340034003000300034003400300030003400340030003000340034003000300034003400300030003400340030003000340034003000300034003400300030003400340030"> : tensor<64x2x1x1xf16>
    %10 = coreai.constant dense<0> : tensor<64x2x1x1xsi8>
    %11 = coreai.constant dense<1> : tensor<si32>
    %12 = coreai.quantize %arg0, %5, %6, %4, %11 : (tensor<1x64x1x64xf16>, tensor<64xf16>, tensor<64xsi8>, tensor<64xf16>, tensor<si32>) -> tensor<1x64x1x64xsi8>
    %13 = coreai.dequantize %12, %5, %6, %4, %11 : (tensor<1x64x1x64xsi8>, tensor<64xf16>, tensor<64xsi8>, tensor<64xf16>, tensor<si32>) -> tensor<1x64x1x64xf16>
    %14 = coreai.lut_to_dense %7, %8, %3 : (tensor<64x64x1x1xui4>, tensor<1x1x1x1x16x1xsi8>, tensor<si16>) -> tensor<64x64x1x1xsi8>
    %15 = coreai.blockwise_shift_scale %14, %9, %10, %2 : (tensor<64x64x1x1xsi8>, tensor<64x2x1x1xf16>, tensor<64x2x1x1xsi8>, tensor<64x2x1x1xf16>) -> tensor<64x64x1x1xf16>
    %16 = coreai.conv2d %13, %15, %1, %1, %0 : (tensor<1x64x1x64xf16>, tensor<64x64x1x1xf16>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>) -> tensor<1x64x1x64xf16>
    coreai.output %16 : tensor<1x64x1x64xf16>
  }
}
