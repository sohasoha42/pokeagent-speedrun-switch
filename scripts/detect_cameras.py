from pokeagent_speedrun_switch.camera import detect_cameras

for probe in detect_cameras():
    print(probe.index, probe.opened)
    if probe.opened:
        print("  read:", probe.readable, probe.shape)
    
