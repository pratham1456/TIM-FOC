import depthai as dai

devices = dai.Device.getAllAvailableDevices()

if not devices:
    print("No OAK devices found")
else:
    print("Available OAK devices:")
    for d in devices:
        print(f" - {d.getMxId()}  ({d.name})")
