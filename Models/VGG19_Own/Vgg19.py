import sys
sys.path.append('D:/Research Paper Visualizations/PlotNeuralNet')
from pycore.tikzeng import *
from pycore.blocks import *

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

       #input
    to_input( '../examples/fcn8s/cats.jpg' ),

    # Block 1
    *block_2ConvPool(name='b1', botton='input', top='pool_b1', s_filer=224, n_filer=64, offset="(0,0,0)", size=(40,40,2.5), opacity=0.5),
    
    # Block 2
    *block_2ConvPool(name='b2', botton='pool_b1', top='pool_b2', s_filer=112, n_filer=128, offset="(1,0,0)", size=(32,32,3.5), opacity=0.5),
    
    # Block 3
    *block_4ConvPool(name='b3', botton='pool_b2', top='pool_b3', s_filer=56, n_filer=256, offset="(1,0,0)", size=(25,25,4.5), opacity=0.5),
    
    # Block 4
    *block_4ConvPool(name='b4', botton='pool_b3', top='pool_b4', s_filer=28, n_filer=512, offset="(1,0,0)", size=(16,16,5.5), opacity=0.5),
    
    # Block 5
    *block_4ConvPool(name='b5', botton='pool_b4', top='pool_b5', s_filer=14, n_filer=512, offset="(1,0,0)", size=(16,16,5.5), opacity=0.5),
    
    # Flatten Layer
    to_FullyConnected(name='flatten', s_filer=7, offset="(2,0,0)", to="(pool_b5-east)", width=1, height=1, depth=40, caption='Flatten'),
    to_connection('pool_b5', 'flatten'),
    
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
