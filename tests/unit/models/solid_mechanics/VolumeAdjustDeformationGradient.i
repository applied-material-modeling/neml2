
[Drivers]
    [unit]
        type = ModelUnitTest
        model = 'model'
        input_R2_names = 'state/F'
        input_R2_values = 'input'
        input_Scalar_names = 'state/J'
        input_Scalar_values = 0.5
        output_R2_names = 'state/Fe'
        output_R2_values = 'output'
    []
[]

[Tensors]
    [output]
        type = FillR2
        values = '2.6458342048  1.5119052599  4.6617078846  
                  5.6696447245  8.5674631393  9.9533812942  
                  2.0158736799  3.1498026248  2.8978184148'
    []
    [input]
        type = FillR2
        values = '2.1 1.2 3.7
                  4.5 6.8 7.9
                  1.6 2.5 2.3 '
    []
[]

[Models]
    [model]
        type = VolumeAdjustDeformationGradient
        input = 'state/F'
        output = 'state/Fe'
        jacobian = 'state/J'
    []
[]
  