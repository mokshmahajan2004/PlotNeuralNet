
from .tikzeng import *

#define new block
def block_2ConvPool( name, botton, top, s_filer=256, n_filer=64, offset="(1,0,0)", size=(32,32,3.5), opacity=0.5 ):
    return [
    to_ConvConvRelu( 
        name="ccr_{}".format( name ),
        s_filer=str(s_filer), 
        n_filer=(n_filer,n_filer), 
        offset=offset, 
        to="({}-east)".format( botton ), 
        width=(size[2],size[2]), 
        height=size[0], 
        depth=size[1],   
        ),    
    to_Pool(         
        name="{}".format( top ), 
        offset="(0,0,0)", 
        to="(ccr_{}-east)".format( name ),  
        width=1,         
        height=size[0] - int(size[0]/4), 
        depth=size[1] - int(size[0]/4), 
        opacity=opacity, ),
    to_connection( 
        "{}".format( botton ), 
        "ccr_{}".format( name )
        )
    ]


def block_Unconv( name, botton, top, s_filer=256, n_filer=64, offset="(1,0,0)", size=(32,32,3.5), opacity=0.5 ):
    return [
        to_UnPool(  name='unpool_{}'.format(name),    offset=offset,    to="({}-east)".format(botton),         width=1,              height=size[0],       depth=size[1], opacity=opacity ),
        to_ConvRes( name='ccr_res_{}'.format(name),   offset="(0,0,0)", to="(unpool_{}-east)".format(name),    s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1], opacity=opacity ),       
        to_Conv(    name='ccr_{}'.format(name),       offset="(0,0,0)", to="(ccr_res_{}-east)".format(name),   s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1] ),
        to_ConvRes( name='ccr_res_c_{}'.format(name), offset="(0,0,0)", to="(ccr_{}-east)".format(name),       s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1], opacity=opacity ),       
        to_Conv(    name='{}'.format(top),            offset="(0,0,0)", to="(ccr_res_c_{}-east)".format(name), s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1] ),
        to_connection( 
            "{}".format( botton ), 
            "unpool_{}".format( name ) 
            )
    ]

def block_4ConvPool(name, botton, top, s_filer, n_filer, offset, size, opacity):
    return [
        to_ConvConvRelu(name=f'ccr_{name}1', s_filer=s_filer, n_filer=(n_filer, n_filer), offset=offset, to=f'({botton}-east)', width=(size[2], size[2]), height=size[0], depth=size[1]),
        to_ConvConvRelu(name=f'ccr_{name}2', s_filer=s_filer, n_filer=(n_filer, n_filer), offset="(0,0,0)", to=f'(ccr_{name}1-east)', width=(size[2], size[2]), height=size[0], depth=size[1]),
        to_ConvConvRelu(name=f'ccr_{name}3', s_filer=s_filer, n_filer=(n_filer, n_filer), offset="(0,0,0)", to=f'(ccr_{name}2-east)', width=(size[2], size[2]), height=size[0], depth=size[1]),
        to_ConvConvRelu(name=f'ccr_{name}4', s_filer=s_filer, n_filer=(n_filer, n_filer), offset="(0,0,0)", to=f'(ccr_{name}3-east)', width=(size[2], size[2]), height=size[0], depth=size[1]),
        to_Pool(name=top, offset="(0,0,0)", to=f'(ccr_{name}4-east)', width=1, height=size[0] * 0.8, depth=size[1] * 0.8, opacity=opacity),
    ]

def block_Dense(name, botton, top, s_filer=256, n_filer=64, num_layers=6, growth_rate=32, offset="(1,0,0)", size=(32,32,3.5), opacity=0.5):
    layers = []
    last_layer = botton  # Start with the input layer

    for i in range(num_layers):
        layer_name = f"{name}_layer{i+1}"
        
        # Add a convolutional layer
        layers.append(
            to_Conv(
                name=layer_name,
                offset=offset,
                to=f"({last_layer}-east)",
                s_filer=str(s_filer),
                n_filer=str(n_filer + i * growth_rate),  # Filters increase with growth rate
                width=size[2],
                height=size[0],
                depth=size[1]
            )
        )
        
        # Create direct connections from previous layers to simulate feature concatenation
        for j in range(i + 1):  # Connect all previous layers
            layers.append(to_connection(f"{name}_layer{j+1}" if j > 0 else botton, layer_name))

        last_layer = layer_name  # Update last_layer

    # Connect all layers to the transition layer
    for i in range(num_layers):
        layers.append(to_connection(f"{name}_layer{i+1}", top))

    return layers



def block_Res( num, name, botton, top, s_filer=256, n_filer=64, offset="(0,0,0)", size=(32,32,3.5), opacity=0.5 ):
    lys = []
    layers = [ *[ '{}_{}'.format(name,i) for i in range(num-1) ], top]
    for name in layers:        
        ly = [ to_Conv( 
            name='{}'.format(name),       
            offset=offset, 
            to="({}-east)".format( botton ),   
            s_filer=str(s_filer), 
            n_filer=str(n_filer), 
            width=size[2],
            height=size[0],
            depth=size[1]
            ),
            to_connection( 
                "{}".format( botton  ), 
                "{}".format( name ) 
                )
            ]
        botton = name
        lys+=ly
    
    lys += [
        to_skip( of=layers[1], to=layers[-2], pos=1.25),
    ]
    return lys


