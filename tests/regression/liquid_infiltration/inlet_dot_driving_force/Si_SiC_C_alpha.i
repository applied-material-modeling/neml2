nbatch = '(1)'

[Tensors]
    ############### Run condition ############### 
    [times]
        type = ScalarFromTorchScript
        pytorch_pt_file = 'aLIndot.pt'
        tensor_name = 'time'
    []
    [aLInDot]
        type = ScalarFromTorchScript
        pytorch_pt_file = 'aLIndot.pt'
        tensor_name = 'data'
    []
    ############### Simulation parameters ############### 
    [M]
        type = Scalar
        values = '0.576'
        batch_shape = '${nbatch}'
    []
    [phi0]
        type = Scalar
        values = '0.5'
        batch_shape = '${nbatch}'
    []
    [omega_L]
        type = Scalar
        values = 1.2e-5
        batch_shape = '${nbatch}'
    []
    [omega_P]
        type = Scalar
        values = 1.25e-5
        batch_shape = '${nbatch}'
    []
    [p]
        type = Scalar
        values = '1.0'
        batch_shape = '${nbatch}'
    []
    [lc]
        type = Scalar
        values = '2.236'
        batch_shape = '${nbatch}'
    []
[]

[Drivers]
    [regression]
      type = LiquidInfiltrationDriver
      model = 'model'
      prescribed_time = 'times'
      time = 'forces/tt'

      prescribed_liquid_inlet_rate = 'aLInDot'
      liquid_inlet_rate = 'forces/aLInDot'
      
      show_input_axis = true
      show_output_axis = true
      show_parameters = false
      ic_Scalar_names = 'state/alpha state/delta state/h state/alphaP'
      ic_Scalar_values = '1e-4 1e-2 1e-3 1e-3'
      save_as = 'test.pt'

      verbose = true
    []
  []

[Solvers]
    [newton]
        type = NewtonWithLineSearch
        linesearch_type = STRONG_WOLFE
        rel_tol = 1e-8
        abs_tol = 1e-10
        max_its = 100
        verbose = false
    []
[]

[Models]
    [inlet_gap]
        type = InletGap
        product_thickness_growth_ratio = 'M'
        initial_porosity = 'phi0'
        product_thickness = 'state/delta'
        inlet_gap = 'state/r1'
    []
    [hdot]
        type = ScalarVariableRate
        variable = 'state/h'
        time = 'forces/tt'
        rate = 'state/hdot'
    []
    [hL]
        type = LiquidSpan
        liquid_molar_volume = 'omega_L'
        inlet_gap = 'state/r1'
        liquid_saturation = 'state/alpha'
        liquid_span = 'state/hL'
    []
    [pcond]
        type = ScalarLinearCombination
        coefficients = "1.0 -1.0"
        from_var = 'state/h state/hL'
        to_var = 'state/pcond'
    []
    [fbcond]
        type = FischerBurmeister
        condition_a = 'state/pcond'
        condition_b = 'state/hdot'
        fischer_burmeister = 'residual/h'
    []
    ############### H RESIDUAL ############### 
    [residual_h]
        type = ComposedModel
        models = 'fbcond inlet_gap hL pcond hdot'
    []
    #############################################
    [product_geo]
        type = ProductSaturation
        product_molar_volume = 'omega_P'
        product_span = 'state/h'
        inlet_gap = 'state/r1'
        product_thickness = 'state/delta'
        product_saturation = 'state/alphaP'
    []
    [alpha_transition]
        type = SwitchingFunction
        smoothness = 100.0
        smooth_type = 'SIGMOID'
        scale = 1.0
        offset = 1.0
        complement_condition = true
        variable = 'state/hL'
        out = 'state/alpha_transition'
    []
    [aR_dot]
        type = ScalarVariableRate
        variable = 'state/alphaP'
        time = 'forces/tt'
        rate = 'state/aRdot'
    []
    [alpha_dot]
        type = ScalarVariableRate
        variable = 'state/alpha'
        time = 'forces/tt'
        rate = 'state/alphadot'
    []
    [mass_balance]
        type = LIMassBalance
        in = 'forces/aLInDot'
        switch = 'state/alpha_transition'
        minus_reaction = 'state/aRdot'
        stoichiometric_coefficient = 1.0
        current = 'state/alphadot'
        total = 'residual/alpha'
    []
    ############### ALPHA RESIDUAL ############### 
    [residual_alpha]
        type = ComposedModel
        models = 'product_geo inlet_gap alpha_transition aR_dot alpha_dot mass_balance hL'
    []
    #############################################
    [deficient_scale]
        type = PowerLawLiquidDeficiency
        product_span = 'state/h'
        liquid_span = 'state/hL'
        exponent = 'p'
        scale = 'state/def_scale'
    []
    [perfect_growth]
        type = DiffusionalProductThicknessGrowth
        liquid_product_density_ratio = 0.8
        initial_porosity = 'phi0'
        product_thickness_growth_ratio = 'M'
        liquid_product_diffusion_coefficient = 5e-7
        representative_pores_size = 'lc'

        inlet_gap = 'state/r1'
        product_thickness = 'state/delta'
        ideal_thickness_growth = 'state/delta_growth'
    []
    [delta_dcrit_ratio]
        type = ProductThicknessLimitRatio
        initial_porosity = 'phi0'
        product_thickness_growth_ratio = 'M'
        product_thickness = 'state/delta'
        limit_ratio = 'state/dratio'
    []
    [delta_limit]
        type = SwitchingFunction
        smoothness = 100.0
        smooth_type = 'SIGMOID'
        scale = 1.0
        offset = 1.0
        complement_condition = true
        variable = 'state/dratio'
        out = 'state/dlimit'
    []
    [ddot]
        type = ScalarVariableRate
        variable = 'state/delta'
        time = 'forces/tt'
        rate = 'state/ddot'
    []
    [product_thickness_growth]
        type = ProductThicknessGrowthRate
        thickness_rate = 'state/ddot'
        scale = 'state/def_scale'
        ideal_thickness_growth = 'state/delta_growth'
        switch = 'state/dlimit'
        residual_delta = 'residual/delta'
    []
    ############### DELTA RESIDUAL ############### 
    [residual_delta]
        type = ComposedModel
        models = 'deficient_scale inlet_gap perfect_growth delta_dcrit_ratio delta_limit ddot product_thickness_growth hL'
    []
    #############################################
    [model_residual]
        type = ComposedModel
        models = 'residual_h residual_alpha residual_delta'
        automatic_scaling = true
    []
    [model_update]
        type = ImplicitUpdate
        implicit_model = 'model_residual'
        solver = 'newton'
    []
    [aSiC_new]
        type = ProductSaturation
        product_molar_volume = 'omega_P'
        product_span = 'state/h'
        inlet_gap = 'state/r1'
        product_thickness = 'state/delta'
        product_saturation = 'state/alphaP'
    []
    [model]
        type = ComposedModel
        models = 'model_update inlet_gap aSiC_new'
        additional_outputs = 'state/delta state/h'
    []
[]