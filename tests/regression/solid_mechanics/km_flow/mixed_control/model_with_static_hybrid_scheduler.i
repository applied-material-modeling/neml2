!include model.i

[Schedulers]
  [scheduler]
    type = StaticHybridScheduler
    devices = 'cuda:0 cpu'
    batch_sizes = '7 2'
  []
[]

[Drivers]
  [driver]
    scheduler = scheduler
    async_dispatch = true
  []
[]
