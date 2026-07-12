"""Install stop signals and execute the standalone training CLI."""

from server.training import TrainingStopRequest, training_stop_signals

stop_request = TrainingStopRequest()
with training_stop_signals(stop_request):
    from server.training_cli.cli import main

    main(stop_request=stop_request)
