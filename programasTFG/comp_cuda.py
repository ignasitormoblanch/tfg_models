
import sys, torch, platform, os
print("Python  :", sys.executable)
print("Torch    :", torch.__version__)
print("Wheel CU :", torch.version.cuda)        # debería decir '12.1' o '11.8', nunca None
print("GPU OK   :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Nombre  :", torch.cuda.get_device_name(0))