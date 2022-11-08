from torchsummary import summary
from branch_model import CustomModel

model = CustomModel('MiT-B3')

print(summary(model, (input_shape)))