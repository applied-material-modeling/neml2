nbatch = '(1)'
nstep = 100

# reaction type
Ea = 177820 # J mol-1
A = 5.24e12 # s-1
R = 8.31446261815324 # JK-1mol-1

Y = 0.7
order = 1.0
k = 1.0

# initial mass fraction
ws0 = 0.02 
wb0 = 0.5
wp0 = 0.3

alpha0 = 3.333333 # 1/(1-Y)

[Tensors]
    ############### Run condition ############### 
    [endtime]
        type = Scalar
        values = '100'
        batch_shape = '${nbatch}'
    []
    [times]
        type = LinspaceScalar
        start = 0
        end = endtime
        nstep = '${nstep}'
    []
    [endT]
        type = Scalar
        values = '1500'
        batch_shape = '${nbatch}'
    []
    [T]
        type = LinspaceScalar
        start = 0
        end = endT
        nstep = '${nstep}'
    []
    ############### Simulation parameters ############### 
    [Ea]
        type = Scalar
        values = '${Ea}'
        batch_shape = '${nbatch}'
    []
    [A]
        type = Scalar
        values = '${A}'
        batch_shape = '${nbatch}'
    []
    [order]
        type = Scalar
        values = '${order}'
        batch_shape = '${nbatch}'
    []
    [k]
        type = Scalar
        values = '${k}'
        batch_shape = '${nbatch}'
    []
[]

[Drivers]
    [driver]
        type = TransientDriver
        model = 'model'
        prescribed_time = 'times'
        time = 'forces/tt'

        force_Scalar_names = 'forces/T'
        force_Scalar_values = 'T'
        
        show_input_axis = true
        show_output_axis = true
        show_parameters = true
        
        ic_Scalar_names = 'state/wp state/wb state/ws state/alpha'
        ic_Scalar_values = '${wp0} ${wb0} ${ws0} ${alpha0}'
        save_as = 'test.pt'

        verbose = false
    []
    [regression]
        type = TransientRegression
        driver = 'driver'
        reference = 'gold/result.pt'
    []
[]

[Solvers]
    [newton]        
        type = Newton
        verbose = false
    []
[]

[Models]
    [amount]
        type = PyrolysisConversionAmount
        initial_solid_mass_fraction = '${ws0}'
        initial_binder_mass_fraction = '${wb0}'
        reaction_yield = '${Y}'

        solid_mass_fraction = 'state/ws'
        reaction_amount = 'state/alpha'
    []
    [reaction]
        type = ChemicalReactionMechanism
        scaling_constant = 'k'
        reaction_order = 'order'
        reaction_amount = 'state/alpha'
        reaction_out = 'state/f'
    []
    [pyrolysis]
        type = PyrolysisKinetics
        kinetic_constant = 'A'
        activation_energy = 'Ea'
        ideal_gas_constant = '${R}'
        temperature = 'forces/T'
        reaction = 'state/f'
        out = 'state/pyro'
    []
    [amount_rate]
        type = ScalarVariableRate
        variable = 'state/alpha'
        time = 'forces/tt'
        rate = 'state/alpha_dot'
    []
    [residual_ms]
        type = ScalarLinearCombination
        coefficients = "1.0 -1.0"
        from_var = 'state/alpha_dot state/pyro'
        to_var = 'residual/ws'   
    []
    [rms]
        type = ComposedModel
        models = 'amount reaction pyrolysis amount_rate residual_ms'
    []
    [solid_rate]
        type = ScalarVariableRate
        variable = 'state/ws'
        time = 'forces/tt'
        rate = 'state/ws_dot'
    []
    [binder_rate]
        type = ScalarVariableRate
        variable = 'state/wb'
        time = 'forces/tt'
        rate = 'state/wb_dot'
    []
    [residual_binder]
        type = ScalarLinearCombination
        coefficients = "${Y} 1.0"
        from_var = 'state/wb_dot state/ws_dot'
        to_var = 'residual/wb'
    []
    [rmb]
        type = ComposedModel
        models = 'solid_rate binder_rate residual_binder'
    []
    [rmp]
        type = ScalarLinearCombination
        coefficients = "1.0 -1.0"
        from_var = 'state/wp old_state/wp'
        to_var = 'residual/wp'
    []
    [model_residual]
        type = ComposedModel
        models = 'rmp rms rmb'
        automatic_scaling = false
    []
    [model_update]
        type = ImplicitUpdate
        implicit_model = 'model_residual'
        solver = 'newton'
    []
    [amount_new]
        type = PyrolysisConversionAmount
        initial_solid_mass_fraction = '${ws0}'
        initial_binder_mass_fraction = '${wb0}'
        reaction_yield = '${Y}'

        solid_mass_fraction = 'state/ws'
        reaction_amount = 'state/alpha'
    []
    [model]
        type = ComposedModel
        models = 'amount_new model_update'
        additional_outputs = 'state/ws'
    []
[]