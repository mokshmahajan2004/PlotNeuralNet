import sys
sys.path.append('D:/Research Paper Visualizations/PlotNeuralNet')
from pycore.tikzeng import *
from pycore.blocks import *

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # Input Image
    to_input('drowsy.png', width=10, height=10, name='input'),

    # Initial Convolution Layer
   to_ConvConvRelu(
    name='conv1',
    s_filer=224,
    n_filer=[64, 64],  # Keep this as a list or tuple
    offset="(0,0,0)",
    to="(input-east)",
    width=[2, 2],  # Make sure this is a list or tuple
    height=40,
    depth=40,
    caption="Conv1"),
    to_connection("input", "conv1"),

    # Dense Block 1
    *block_2ConvPool(name='db1', botton='conv1', top='pool_db1', s_filer=112, n_filer=128, offset="(1,0,0)", size=(32,32,3.5), opacity=0.5),

    # Dense Block 2
    *block_2ConvPool(name='db2', botton='pool_db1', top='pool_db2', s_filer=56, n_filer=256, offset="(1,0,0)", size=(25,25,4.5), opacity=0.5),

    # Dense Block 3
    *block_2ConvPool(name='db3', botton='pool_db2', top='pool_db3', s_filer=28, n_filer=512, offset="(1,0,0)", size=(16,16,5.5), opacity=0.5),

    # Dense Block 4
    *block_2ConvPool(name='db4', botton='pool_db3', top='pool_db4', s_filer=14, n_filer=1024, offset="(1,0,0)", size=(12,12,6.5), opacity=0.5),

    # Flatten Layer
    to_FullyConnected(name='flatten', s_filer=7, offset="(2,0,0)", to="(pool_db4-east)", width=1, height=1, depth=40, caption='Flatten'),
    to_connection('pool_db4', 'flatten'),

    # Fully Connected Layers
    to_FullyConnected(name='fc1', s_filer=1, offset="(1,0,0)", to="(flatten-east)", width=1, height=1, depth=30, caption='Dense 512'),
    to_connection('flatten', 'fc1'),

    to_FullyConnected(name='fc2', s_filer=1, offset="(1,0,0)", to="(fc1-east)", width=1, height=1, depth=20, caption='Dense Softmax'),
    to_connection('fc1', 'fc2'),

    to_SoftMax(name='softmax', s_filer=5, offset="(1,0,0)", to="(fc2-east)", caption='Output'),
    to_connection('fc2', 'softmax'),

    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()
