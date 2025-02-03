nbatch = '(1)'
omega_L = 1.2e-5

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
        values = '0.8'
        batch_shape = '${nbatch}'
    []
    [phi0]
        type = Scalar
        values = '0.9'
        batch_shape = '${nbatch}'
    []
    [omega_L]
        type = Scalar
        values = '${omega_L}'
        batch_shape = '${nbatch}'
    []
    [omega_P]
        type = Scalar
        values = 1.25e-5
        batch_shape = '${nbatch}'
    []
    [lc]
        type = Scalar
        values = '0.5'
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
      show_parameters = true
      ic_Scalar_names = 'state/alpha state/delta state/alphaP state/h'
      ic_Scalar_values = '1.0 1e-2 1e-3 1.0'
      save_as = 'test.pt'

      verbose = true
    []
  []

[Solvers]
    [newton]
        type = NewtonWithLineSearch
        linesearch_type = STRONG_WOLFE
        linesearch_cutback = 1.55
        
        #type = Newton
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
    #############################################
    [product_geo]
        type = ProductSaturation
        product_molar_volume = 'omega_P'
        product_span = 'state/h'
        maximum_span_condition = true
        inlet_gap = 'state/r1'
        product_thickness = 'state/delta'
        product_saturation = 'state/alphaP'
    []
    [alphamax]
        type = LiquidSaturationLimitRatio
        liquid_molar_volume = 'omega_L'
        inlet_gap = 'state/r1'
        liquid_saturation = 'state/alpha'
        limit_saturation_ratio = 'state/alphamax'
    []
    [alpha_transition]
        type = SwitchingFunction
        smoothness = 100.0
        smooth_type = 'SIGMOID'
        scale = 1.0
        offset = 1.0
        complement_condition = true
        variable = 'state/alphamax'

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
        models = 'product_geo inlet_gap alpha_transition aR_dot alpha_dot mass_balance alphamax'
    []
    #############################################
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
    [alphalowbound]
        type = SwitchingFunction
        smoothness = 100.0
        smooth_type = 'SIGMOID'
        scale = 1.0
        offset = 0.0
        complement_condition = false
        variable = 'state/alphamax'
        out = 'state/alphazero'
    []
    [product_thickness_growth]
        type = ProductThicknessGrowthRate
        thickness_rate = 'state/ddot'
        scale = 'state/alphazero'
        #scaling_condition = false
        ideal_thickness_growth = 'state/delta_growth'
        switch = 'state/dlimit'
        residual_delta = 'residual/delta'
    []
    ############### DELTA RESIDUAL ############### 
    [residual_delta]
        type = ComposedModel
        models = 'inlet_gap perfect_growth delta_dcrit_ratio delta_limit ddot product_thickness_growth alphalowbound alphamax'
    []
    #############################################
    [dummy]
        type = ScalarLinearCombination
        coefficients = "1.0 -1.0"
        from_var = 'state/h old_state/h'
        to_var = 'residual/h'
    []
    [model_residual]
        type = ComposedModel
        models = 'residual_alpha residual_delta dummy'
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
        maximum_span_condition = true
        inlet_gap = 'state/r1'
        product_thickness = 'state/delta'
        product_saturation = 'state/alphaP'
    []
    [diffusion_coef]
        type = EffectiveDiffusivity
        macroscopic_diffusion_coefficient = 1.2e-3
        coefficient_liquid_limit = 100.0
        scaling_liquid_limit = 'state/delta'
        const_liquid_limit = true
        switching_liquid_limit = 'state/alpha_transition'
        effective_diffusion_coefficient = 'state/Deff'
    []
    [model]
        type = ComposedModel
        models = 'model_update inlet_gap aSiC_new alpha_transition diffusion_coef'
        additional_outputs = 'state/delta state/h'
    []
[]