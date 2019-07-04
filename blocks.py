import layers

def down(input, dim_out, name, size=3, stride=1):
    c1 =  layers.conv_layer(input, dim_out, size, stride, name+'_c1')
    c2 = layers.conv_layer(c1, dim_out, size, stride, name+'_c2')
    mp = layers.avgpool_layer(c2, name)

    # print(c2, mp)

    return c2, mp

def up(input, skip, dim_out, name, up_size=2, up_stride=2, conv_size=3, conv_stride=1):
    up = layers.up_layer(input, dim_out, up_size, up_stride, name)
    concat = layers.concat_layer(up, skip, name)
    c1 = layers.conv_layer(concat, dim_out, conv_size, conv_stride, name+'_c1')
    c2 = layers.conv_layer(c1, dim_out, conv_size, conv_stride, name+'_c2')

    # print(c2)

    return c2

def bottleneck(input, dim_out, name, size=3, stride=1):
    c1 = layers.conv_layer(input, dim_out, size, stride, name+'_c1')
    c2 = layers.conv_layer(c1, dim_out, size, stride, name+'_c2')

    # print(c2)

    return c2

def res_down(input, dim_out, name, size=3, stride=1):
    c1 =  layers.conv_layer(input, dim_out, size, stride, name+'_c1')
    c2 = layers.conv_layer(c1, dim_out, size, stride, name+'_c2')

    shortcut = layers.conv1x1_layer(input, dim_out, name+'_shortcut')
    add = shortcut + c2

    mp = layers.avgpool_layer(add, name)

    # print(c2, mp)

    return c2, mp

def res_up(input, skip, dim_out, name, up_size=2, up_stride=2, conv_size=3, conv_stride=1):
    up = layers.up_layer(input, dim_out, up_size, up_stride, name)
    concat = layers.concat_layer(up, skip, name)
    c1 = layers.conv_layer(concat, dim_out, conv_size, conv_stride, name+'_c1')
    c2 = layers.conv_layer(c1, dim_out, conv_size, conv_stride, name+'_c2')

    shortcut = layers.conv1x1_layer(concat, dim_out, name+'_shortcut')
    add = shortcut + c2

    # print(c2)

    return add
