!include model.i

[Schedulers]
  [scheduler]
    type = SimpleScheduler
    device = cpu
    batch_size = 8
  []
[]

[Drivers]
  [driver]
    scheduler = scheduler
    async_dispatch = false
  []
[]
