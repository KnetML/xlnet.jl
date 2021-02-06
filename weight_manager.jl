using FileIO

atype = Array{Float32}

function create_single_layer_weight_dict()
    l = Dict()
    l["post_attn"] = Dict()
    l["rel_attn"] = Dict()
    l["ff"] = Dict()

    l["post_attn"]["layer_norm"] = Dict()
    l["ff"]["layer_1"] = Dict()
    l["ff"]["layer_2"] = Dict()
    l["ff"]["layer_norm"] = Dict()
    l
end

function create_weight_dict(n)
    w = Dict()
    for i=0:n-1
        w[ "layer_"*string(i) ]=create_single_layer_weight_dict()
    end
    w
end

function load_layer_from_model( layer )
    l = create_single_layer_weight_dict()
    l["post_attn"]["o"] = atype( value( layer.post_attention.proj_o ) )
    l["post_attn"]["layer_norm"]["gamma"] = atype( value( layer.post_attention.layer_norm.gamma ) )
    l["post_attn"]["layer_norm"]["beta"] = atype(value( layer.post_attention.layer_norm.beta ) )

    l["rel_attn"]["k"] = atype(value( layer.head_proj.k_proj_weight ) ) 
    l["rel_attn"]["q"] = atype(value( layer.head_proj.q_proj_weight ) )
    l["rel_attn"]["v"] = atype(value( layer.head_proj.v_proj_weight ) )
    l["rel_attn"]["r"] = atype(value( layer.head_proj.r_proj_weight ) )

    l["ff"]["layer_1"]["kernel"] = atype(value( layer.ffn.layer1.kernel ) )
    l["ff"]["layer_1"]["bias"] = atype(value( layer.ffn.layer1.bias ) )

    l["ff"]["layer_2"]["kernel"] = atype(value( layer.ffn.layer2.kernel ) )
    l["ff"]["layer_2"]["bias"] = atype(value( layer.ffn.layer2.bias ) )

    l["ff"]["layer_norm"]["gamma"] = atype( value( layer.ffn.layer_norm.gamma ) )
    l["ff"]["layer_norm"]["beta"] = atype( value( layer.ffn.layer_norm.beta ) )
    l
end
    
function save_weights(path,model)
    w = create_weight_dict(model.n_layer)
    for (i,layer) in enumerate(model.layers)
        println("i = " ,i)
        w[ "layer_" * string(i-1) ] = load_layer_from_model( layer )
    end
    
    w["word_emb"] = value( model.embedding.lookup_table )
    w["r_w_bias"] = value( model.r_w_bias )
    w["r_r_bias"] = value( model.r_r_bias )
    w["r_s_bias"] = value( model.r_s_bias )
    w["seg_emb"] = value( model.seg_embed )
    w["mask_emb"] = value( model.mask_emb )
    save(path,w)

end

function save_p_weights(path,p)
    m = p.model
    wm = create_weight_dict(model.n_layer)

    for (i,layer) in enumerate(model.layers)
        println("i = " ,i)
        wm[ "layer_" * string(i-1) ] = load_layer_from_model( layer )
    end
    
    wm["word_emb"] = atype( value( model.embedding.lookup_table ) )
    wm["r_w_bias"] = atype( value( model.r_w_bias ) )
    wm["r_r_bias"] = atype( value( model.r_r_bias ) )
    wm["r_s_bias"] = atype( value( model.r_s_bias ) )
    wm["seg_emb"] =  atype( value( model.seg_embed ) )
    wm["mask_emb"] = atype( value( model.mask_emb ) )

    wp = Dict()
    wp["model"] = wm
    wp["w"] = atype( value(p.w) )
    wp["b"] = atype( value(p.b) )
    wp["projection"] = Dict()
    wp["projection"]["w"] = atype( value(p.projection.w) )
    wp["projection"]["b"] = atype( value(p.projection.b) )
 
    save(path,wp)
end

function get_linear(l)
    d = Dict()
    d["w"] = atype( value(l.w) )
    d["b"] = atype( value(l.b) )
    d
end


function save(path::String, p::XLNetClassifier)
    m = p.model
    wm = create_weight_dict(model.n_layer)

    for (i,layer) in enumerate(model.layers)
        println("i = " ,i)
        wm[ "layer_" * string(i-1) ] = load_layer_from_model( layer )
    end
    
    wm["word_emb"] = atype( value( model.embedding.lookup_table ) )
    wm["r_w_bias"] = atype( value( model.r_w_bias ) )
    wm["r_r_bias"] = atype( value( model.r_r_bias ) )
    wm["r_s_bias"] = atype( value( model.r_s_bias ) )
    wm["seg_emb"] =  atype( value( model.seg_embed ) )
    wm["mask_emb"] = atype( value( model.mask_emb ) )

    wp = Dict()
    wp["model"] = wm
    wp["projection"] = get_linear( p.projection )
    wp["classification"] = Dict( "w"=> p.classification.w, "b"=>p.classification.b )
    save(path,wp)
end

function load_weights(path)
    w = load(path)
    w
end
