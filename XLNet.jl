#=
Knet implementation of XLNet
author: Arda Goktogan
ardagoktogan@gmail.com
=#
module XLNet


using Statistics
using IterTools:ncycle
using LinearAlgebra
using Knet:dropout,softmax,param
using NPZ
using Knet
using Statistics
using SpecialFunctions
using Knet.Ops21
using Base.Iterators: take, drop, cycle, Stateful
using IterTools: ncycle, takenth, takewhile
using AutoGrad
using Knet.Ops21: gelu
using JLD2
import CUDA

export specaialTokens, xlnet_base_hparams, create_xlnet_model, XLNetModel, XLNetClassifier, prepare_sample, save
 
specaialTokens = Dict( "<SEP>" => 4, "<CLS>" => 3 )
fptype= Float32
atype = KnetArray{ fptype }

mutable struct EmbeddingLookup
  lookup_table #size: (n_tokens, emb_length)
end

function(e::EmbeddingLookup)(inp)
    e.lookup_table[inp,:]
end

function paramf(x;freeze=false)
    fptype= Float32
    atype = KnetArray{ fptype }

    if freeze==true
        return atype(x)
    else
        return Param( atype(x) )
    end

end

mutable struct LayerNorm
    epsilon
    beta
    gamma

    LayerNorm( w::Dict ;freeze=false ) = new( convert( fptype, 1e-12 ),
                                              paramf( w["beta"], freeze=freeze),
                                              paramf( w["gamma"], freeze=freeze) )
  
    LayerNorm( i::Int64 ;freeze=false ) = new( convert( fptype, 1e-12 ),
                                                paramf( randn(i), freeze=freeze),
                                                paramf( randn(i), freeze=freeze) )
end

function (l::LayerNorm)( inputs )
    μ = mean( inputs, dims=3 )
    σ = std( inputs ,mean=μ,dims=3)
    result = ( reshape( l.gamma, 1, 1, :) .* ( ( inputs .- μ ) ./ ( σ .+ l.epsilon ) ) )  .+ reshape( l.beta, 1, 1, :)   
    return result
end

#3D Dense Layer
mutable struct Layer
    kernel
    bias
    Layer(w::Dict;freeze=false)=new( paramf( w["kernel"], freeze=freeze ) , paramf( w["bias"], freeze= freeze ) )
    Layer(i::Int,o::Int;freeze=false) = new( paramf( randn(o,i) , freeze=freeze ), paramf( zeros(o) , freeze=freeze) )
end

function(l::Layer)(inp)
    #inp = einsum("ijk,kl->ijl", inp, l.kernel )
    (i,j,k)  = size(inp)
    inp = reshape(inp,i*j,k)
    inp = inp * l.kernel
    inp = reshape( inp , i,j,:)
    inp = inp .+ reshape(l.bias,1,1,:)
    inp
end

mutable struct FFN
    layer1::Layer
    layer2::Layer
    layer_norm::LayerNorm
    FFN(w::Dict; freeze=false) = new( Layer( w["layer_1"],freeze=freeze ) ,
                                      Layer( w["layer_2"],freeze=freeze ) ,
                                      LayerNorm( w["layer_norm"], freeze = freeze ) )
end


function (ffn::FFN)( inp, i, p_drop = 0.1 )
    out = inp
    out = gelu.( ffn.layer1(out) )
    out = dropout( out, p_drop )
    out = ffn.layer2(out)
    out = dropout( out, p_drop )
    out = ffn.layer_norm( inp + out )
    out
end


mutable struct HeadProjection
    # x_proj_weight : [d_model, n_head, d_head] , where x stands for {q,k,v,r}
    q_proj_weight
    k_proj_weight
    v_proj_weight
    r_proj_weight
    HeadProjection(i,w::Dict;freeze=false) = new( paramf( w["q"] , freeze=freeze ),
                                                  paramf( w["k"] , freeze=freeze ),
                                                  paramf( w["v"] , freeze=freeze ),
                                                  paramf( w["r"] , freeze=freeze ) )
end

mutable struct PostAttn
    proj_o
    layer_norm::LayerNorm
    PostAttn(w::Dict,i; freeze= false) = new( paramf( w["o"],freeze=freeze ),
                                              LayerNorm( w["layer_norm"] , freeze = freeze ) )
  end

mutable struct AttnLayer
    head_proj::HeadProjection
    ffn::FFN
    post_attention::PostAttn

    AttnLayer(i,w::Dict; freeze = false) = new( HeadProjection( i, w["rel_attn"], freeze = freeze),
                                                FFN( w["ff"], freeze= freeze ),
                                                PostAttn( w["post_attn"], i, freeze = freeze) )
    
end

mutable struct XLNetModel
    n_token
    n_layer
    n_head
    d_head
    d_inner
    d_model
    p_drop
    p_dropatt
    attn_type
    bi_data
    clamp_len
    same_length
    reuse_len
    mem_len
    embedding::EmbeddingLookup #trained
    r_w_bias#trained
    r_r_bias#trained
    
    r_s_bias#trained
    seg_embed#trained
    
    layers#trained    
end

"""
pos_seq: Vector
inv_freq: Vector
bsz: batch size

return: Matrix : (length(pos_seq) , bsz , 2*length(inv_freq) )
"""

function positional_embedding( pos_seq, inv_freq; bsz=nothing )
    
    sinusoid_inp = pos_seq .* inv_freq'
    
    pos_emb = cat( sin.(sinusoid_inp) , cos.(sinusoid_inp) , dims = 2 )
    
    pos_emb = reshape(pos_emb,size(pos_emb)[1],1, size(pos_emb)[2])
    
    if bsz != nothing
        pos_emb = repeat( pos_emb, outer = [1,bsz,1] )
    end
    pos_emb
end

function _create_mask( qlen, mlen; dtype=fptype , same_length=false)
  """create causal attention mask."""

    attn_mask = ones( dtype ,(qlen,qlen) )

    mask_u = UpperTriangular( attn_mask ) + zeros(qlen,qlen)
    mask_dia = zeros(qlen,qlen) + Diagonal( attn_mask )
    attn_mask_pad = zeros(dtype,(qlen,mlen))
    ret = cat( dims = 2,attn_mask_pad, mask_u - mask_dia )

    if same_length
        mask_l = LowerTriangular( attn_mask ) + zeros( size(attn_mask)... )
        ret = cat( dims = 2,
                   ret[:, collect(1:qlen) ] + mask_l - mask_dia ,
                   ret[:, collect(end-qlen:end) ] )

    end
    ret

end


function rel_shift(x,klen)
    
    x = permutedims( x, [4,3,2,1])
    x_size = size(x)
    x = reshape(x,x_size[1],x_size[2],x_size[4],x_size[3] )
    x = x[:,:,:,2:end]
    x = reshape(x, x_size[1], x_size[2], x_size[3] - 1, x_size[4] )
    x = x[: , : , 1:klen , :]
    x = permutedims(x , [4,3,2,1])
    x
end

function einsum_4d_v1(a,b)
    #einsum 'ibnd,jbnd->ijbn'
    ii,bi,ni,di = size(a)
    ji,bi,ni,di = size(b)
    a = permutedims(a, [1,4,2,3] )
    b = permutedims(b, [4,1,2,3] )

    c = bmm(a,b) # size is i,j,b,n
    c

end

function einsum_3d_4d(a,b)
    #einsum ibnd , snd -> ibns
    ii,bi,ni,di = size(a)
    si,ni,di   = size(b)
    a = reshape( a,ii*bi,ni,di )
    a = permutedims( a,[1,3,2] )
    b = permutedims( b,[3,1,2] )
    c = bmm(a,b)
    c = reshape(c,ii,bi,si,ni)
    c = permutedims( c, [1,2,4,3])
    c
end

function einsum_4d_v2(a,b)
    #einsum( "ijbs,ibns->ijbn" )
    ii,ji,bi,si = size(a)
    ii,bi,ni,si = size(b)
    a = permutedims(a,[2,4,1,3]) #size(a) = (j,s,i,b)
    b = permutedims(b,[4,3,1,2]) #size(b) = (s,n,i,b)
    c = bmm(a,b)
    c = permutedims(c,[3,1,4,2])
    c
end

function einsum_4d_v3(a,b)
    #einsum("ijbn,jbnd->ibnd")
    ii,ji,bi,ni = size(a) # ijbn
    ji,bi,ni,di = size(b)
    b = permutedims( b, [1,4,2,3]) #jdbn
    c = bmm(a,b) # idbn
    c = permutedims(c,[1,3,4,2]) #ibnd
    c
end

function einsum_3d(a,b)
    #einsum("ijk,klm->ijlm")
    i,j,k = size(a)
    k,l,m = size(b)
    a = reshape(a, i*j, k)
    b = permutedims( b, [3,2,1] )  #size of b is m,l,k
    b = reshape(b, l*m ,k )
    b = permutedims( b, [2,1] )
    c = a*b
    c = reshape( c,i,j,m,l )
    c = permutedims( c , [1,2,4,3] )
    sc = size(c)
    c
end

function einsum_4d_3d(a,b)
    #einsum("ibnd,hnd->ibh")
    ii,bi,ni,di = size(a)
    hi,ni,di  = size(b)
    a = reshape(a,ii*bi,ni*di )
    b = permutedims( b, [2,3,1] )
    b = reshape( b, ni*di, hi )
    c = a*b
    c = reshape(c, ii, bi, hi)
    c
end

function rel_attn_core( q_head, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
                      r_w_bias, r_r_bias, r_s_bias, attn_mask, scale, p_dropatt = 0.1 )

    """Core relative positional attention operations."""
    r_w_bias = reshape( r_w_bias, 1, 1, size( r_w_bias )... )
    r_r_bias = reshape( r_r_bias, 1, 1, size( r_r_bias )... )
    ac = einsum_4d_v1( q_head .+ r_w_bias , k_head_h )
    bd = einsum_4d_v1( q_head .+ r_r_bias , k_head_r )
    bd = rel_shift( bd, size(ac)[2] )

    # segment based attention score    
    if seg_mat==nothing
        ef = 0
    else 
        ef = einsum_3d_4d( q_head .+ reshape( r_s_bias, (1,1,size(r_s_bias)...) ), seg_embed )
        ef = einsum_4d_v2( seg_mat , ef )
    end
    attn_score = (ac .+ bd .+ ef) * scale
    if attn_mask != nothing
        attn_score = attn_score .- fptype(1e9) * attn_mask
    end

    attn_prob = softmax(attn_score, dims=2 )
    attn_prob = dropout( attn_prob, p_dropatt )
    
    attn_vec = einsum_4d_v3( attn_prob, v_head_h )
    attn_vec
end


"""
q_len: sequance length
klen: memory length + sequance length
d_model: hidden size of the model
clamp_length: clamp all relative distances larger than clamp_length
attn_type: attention type (bidirectional as default)
bi_data: bool, wheather to use bidirectional input pipeline

returns: [klen+qlen,1,d_model]
"""

function relative_positional_encoding(qlen, klen, d_model, clamp_len, attn_type,
                                      bi_data , bsz ; dtype = fptype, atype = KnetArray{ fptype } )
  """create relative positional encoding."""
    
    freq_seq = collect( range( 0 , step=2.0 , stop=d_model-1 ) )
    
    if dtype != nothing && dtype != fptype
        freq_seq = convert.( dtype, freq_seq )
    end
    
    inv_freq =  1 ./( 10000 .^ (freq_seq./d_model) ) 
    
    if attn_type == "bi"
        first, last = klen, -qlen + 1
    elseif  attn_type == "uni"
        first, last = klen, 0
    end
    
    if bi_data
        fwd_pos_seq = collect( range( first , step=-1.0 , stop=last ) )
        bwd_pos_seq = collect( range( -first , step=1.0 , stop=-last ) )
        if dtype != nothing # && dtype != Float16
            fwd_pos_seq = convert.( fptype , fwd_pos_seq)
            bwd_pos_seq = convert.( fptype , bwd_pos_seq)
        end
        if clamp_len>0
            fwd_pos_seq = clamp.(fwd_pos_seq , -clamp_len, clamp_len )
            bwd_pos_seq = clamp.(bwd_pos_seq , -clamp_len, clamp_len )
        end
        
        if bsz != nothing
            fwd_pos_emb =  positional_embedding(fwd_pos_seq, inv_freq, bsz = convert(Int8,bsz/2))
            bwd_pos_emb =  positional_embedding(bwd_pos_seq, inv_freq, bsz = convert(Int8,bsz/2))
        else
            fwd_pos_emb =  positional_embedding(fwd_pos_seq, inv_freq)
            bwd_pos_emb =  positional_embedding(bwd_pos_seq, inv_freq)
        end
        pos_emb = cat(dims=2 , fwd_pos_emb , bwd_pos_emb)
        
    else
        fwd_pos_seq = collect(range(first , step=-1.0 , stop=last ) )
        if dtype != nothing
            fwd_pos_seq = convert.( fptype , fwd_pos_seq)
        end
        if clamp_len > 0
            fwd_pos_seq = clamp.(fwd_pos_seq , -clamp_len, clamp_len )
        end

        pos_emb = positional_embedding( fwd_pos_seq , inv_freq )
    end  
    atype( pos_emb )  
end

function cache_mem(curr_out, prev_mem, mem_len, reuse_len=nothing )
    if mem_len != nothing || mem_len == 0
        return nothing
    else
        if reuse_len != nothing && reuse_len > 0
            curr_out = curr[1:reuse_len]
        end
        
        if prev_mem == nothing
            new_mem = curr_out[end-mem_len+1:end]
        else
            new_mem = cat( disms=1,prev_mem,curr_out)[end-mem_len+1:end]
        end
    end
    new_mem
end

function(x::XLNetModel)(inp_k,
                        seg_id,
                        input_mask;
                        mems=nothing,
                        perm_mask=nothing,
                        target_mapping=nothing,
                        inp_q=nothing,
                        attn_type = "bi", dtype = fptype, atype = KnetArray{ fptype } )

    
    new_mems=[]
    bsz = size(inp_k)[2]
    qlen = size(inp_k)[1]
    
    if mems != nothing
        mlen = size( mems[1] )[1]
    else
        mlen=0
    end 
    
    klen = mlen + qlen
    
    #Attention Mask

    if attn_type == "uni"
        attn_mask = _create_mask( qlen, mlen, tf_float, same_length)
        attn_mask = reshape(attn_mask,( 1 , 1 , size(attn_mask)[1] , size(attn_mask)[2] ) )
    elseif attn_type == "bi"
        attn_mask = nothing
    end

    if input_mask != nothing && perm_mask != nothing
        data_mask = reshape( input_mask,( 1,size(input_mask)...) ) + perm_mask
    elseif input_mask != nothing && perm_mask == nothing
        data_mask = reshape(input_mask,( 1,size(input_mask)...) )
    elseif input_mask == nothing && perm_mask != nothing
        data_mask = perm_mask
    else
        data_mask = nothing
    end

    if data_mask != nothing
        # all mems can be attended to
        mems_mask = zeros( dtype ,size(data_mask)[1] , mlen, bsz )
        data_mask = cat( dims=2, mems_mask, data_mask ) 
 
        if attn_mask == nothing
            attn_mask = reshape(data_mask, ( size(data_mask)... , 1  ) )
        else
            attn_mask += reshape(data_mask, ( size(data_mask)... , 1  ) )
        end
    end

    if attn_mask != nothing
        attn_mask = atype( attn_mask )
        attn_mask = 1.0 .* (attn_mask .> 0 )
    end
 
    if attn_mask != nothing
        attn_mask = atype( attn_mask )
        idn = atype( Matrix{fptype}(LinearAlgebra.I, qlen, qlen) )
        non_tgt_mask = -idn
        non_tgt_mask = cat( dims=2 , atype( zeros(dtype,qlen,mlen) ) , non_tgt_mask )
        non_tgt_mask = reshape( non_tgt_mask , size(non_tgt_mask)...,1,1 ) * 1
        non_tgt_mask = attn_mask .+ non_tgt_mask
        non_tgt_mask = atype( 1.0 .* (non_tgt_mask .> 0 ) )
    
    else
        non_tgt_mask = nothing
    end     

    word_emb_k = x.embedding( inp_k .+ 1 )
    output_h = dropout( word_emb_k, x.p_drop )
    ##### Segment embedding
    if seg_id != nothing

        mem_pad = zeros(Int32,mlen,bsz)
        cat_ids = cat(dims=1,mem_pad,seg_id)
        
        seg_mat  = reshape( seg_id, size( seg_id )[1], 1, size( seg_id )[2] ) .== reshape(cat_ids, 1, size(cat_ids)... )
        
        seg_mat = seg_mat*1
        seg_mat_temp = seg_mat
        seg_mat = atype( zeros(qlen,klen,bsz,2) )
        seg_mat[:,:,:,1] = seg_mat_temp[:,:,:]
        seg_mat[:,:,:,2] = 1 .- seg_mat_temp[:,:,:]
        
    else
        seg_mat=nothing
    end
    
    ##### Positional encoding
    pos_emb = relative_positional_encoding(
        qlen, klen, x.d_model, x.clamp_len, attn_type, x.bi_data, bsz )

    #pos_emb = repeat(pos_emb,1,bsz,1)
    pos_emb = dropout(pos_emb, x.p_drop)

    if mems == nothing
        mems = map( x -> nothing, zeros(x.n_layer) )
    end
    
                          
    for (i,attn_layer) in enumerate(x.layers)                                                                                
        if seg_id == nothing
            r_s_bias_i = nothing
            seg_embed_i = nothing
        else
            r_s_bias_i = x.r_s_bias[i,:,:]
            seg_embed_i = x.seg_embed[i,:,:,:]
        end

        output_h = attn_layer(  output_h,
                                pos_emb,
                                x.r_w_bias[i,:,:],
                                x.r_r_bias[i,:,:],
                                seg_mat,
                                r_s_bias_i,
                                seg_embed_i,
                                non_tgt_mask,
                                mems[i],
                                x.d_model,
                                x.n_head,
                                x.d_head,
                                x.p_drop,
                                x.p_dropatt,
                                i)
        
    end
        
    output=output_h
    output
end


function(hp::HeadProjection)(h, name, i )
    """Project hidden states to a specific head with a 4D-shape."""

    proj_weight = nothing
    if( name == 'q' )
        proj_weight = hp.q_proj_weight
    elseif( name == 'k' )
        proj_weight = hp.k_proj_weight
    elseif( name == 'v' )
        proj_weight = hp.v_proj_weight
    elseif( name == 'r' )
        proj_weight = hp.r_proj_weight
    else
        println("unknown name in head_projection")
    end

    head = einsum_3d(h,proj_weight)
    head
end

function (pa::PostAttn)(h,attn_vec,residual=true, p_drop = 0.1 )
    
    attn_out = einsum_4d_3d( attn_vec, pa.proj_o)
    attn_out = dropout( attn_out, p_drop)

    if residual
        output = pa.layer_norm( attn_out + h )
    else
        output = pa.layer_norm( attn_out )
    end
    output

end

function (rma::AttnLayer)(h,r,r_w_bias,r_r_bias,seg_mat, r_s_bias, seg_embed,attn_mask,mems,
        d_model, n_head, d_head,dropout,dropatt, i)
    
    """Multi-head attention with relative positional encoding."""

    scale = convert(fptype,1/sqrt(d_head) )
    if mems != nothing && length( size(mems)) > 1 
        cat = cat(dims=1,mems,h)
    else
        cat = h
    end

    #content heads
    q_head_h = rma.head_proj(h,'q', i)
    k_head_h = rma.head_proj(cat,'k', i)
    v_head_h = rma.head_proj(cat,'v', i)
    
    #positional heads
    k_head_r = rma.head_proj(r,'r', i)

    #core attention ops
    attn_vec = rel_attn_core(q_head_h , k_head_h , v_head_h, k_head_r,seg_embed,seg_mat,r_w_bias, r_r_bias,r_s_bias,attn_mask,scale)

    #post processing
    output = rma.post_attention(h , attn_vec)
    output = rma.ffn(output, i)
    output
end

##--------------------- XLNet Model Settings ----------------------##

xlnet_base_hparams = Dict(  "n_token" => 32000,
                            "n_layer" => 12,
                            "n_freeze" => 10,
                            "n_head" => 12,
                            "d_head" => 64,
                            "d_inner" => 768,
                            "d_model"=>768,
                            "p_drop"=>0.1,
                            "p_dropatt"=>0.1,
                            "attn_type" => "bi",
                            "bi_data" => false,
                            "clamp_len" => -1,
                            "same_length" => false,
                            "reuse_len" => 0,
                            "mem_len" => 0)

function create_xlnet_model( hparam, w )

  n_token     = hparam["n_token"]
  n_layer     = hparam["n_layer"]
  n_freeze    = hparam["n_freeze"]
  n_head      = hparam["n_head"]
  d_head      = hparam["d_head"]
  d_inner     = hparam["d_inner"]
  d_model     = hparam["d_model"]
  p_drop      = hparam["p_drop"]
  p_dropatt   = hparam["p_dropatt"]
  attn_type   = hparam["attn_type"]
  bi_data     = hparam["bi_data"]
  clamp_len   = hparam["clamp_len"]
  same_length = hparam["same_length"]
  reuse_len   = hparam["reuse_len"]
  mem_len     = hparam["mem_len"]
  
  embedding = EmbeddingLookup( paramf( w["word_emb"]  , freeze = true ) )
  r_w_bias = paramf( w["r_w_bias"] , freeze = true )
  r_r_bias = paramf( w["r_r_bias"] , freeze = true )
  r_s_bias = paramf( w["r_s_bias"] , freeze = true )
  seg_embed = paramf( w["seg_emb"] , freeze = true )

  layers = []
  for i in 1:n_layer
      push!( layers,AttnLayer(i, w[ "layer_" * string(i-1) ] , freeze = ( i <= n_freeze ) ) )
  end

  XLNetModel(n_token,
             n_layer,
             n_head,
             d_head,
             d_inner,
             d_model,
             p_drop,
             p_dropatt,
             attn_type,
             bi_data,
             clamp_len,
             same_length,
             reuse_len,
             mem_len,
             embedding,
             r_w_bias,
             r_r_bias,
             r_s_bias,
             seg_embed,
             layers )    
end


##--------------------- CLASSIFIER MODEL FOR DOWNSTREAM TASK ----------------------##

struct Linear;w;b;f;end
(l::Linear)(x) = l.f.( l.w * x .+ l.b )
Linear( w::Dict ) = Linear( paramf( w["w"] ), paramf(w["b"]), w["f"] )

mutable struct XLNetClassifier; model; projection; classification; end

XLNetClassifier( i::Int, o::Int, model ) = XLNetClassifier(model,
                                                           Linear( paramf( xavier(i,i) ), paramf( zeros(i) ), tanh ),
                                                           Linear( paramf( xavier(o,i) ), paramf( zeros(o) ), x->x ) )  

XLNetClassifier( w::Dict, model ) = XLNetClassifier(model,
                                                    Linear(paramf(w["w"]),paramf(w["b"]),tanh),
                                                    Linear(paramf(w["projection"]["w"] ),paramf(w["projection"]["b"] ), x->x )  )

function XLNetClassifier( path::String )
    @load path weights
    w=weights
    model = create_xlnet_model( xlnet_base_hparams, w["model"] )
    XLNetClassifier(model,
                    Linear( paramf( w["projection"]["w"] ), paramf( w["projection"]["b"] ), tanh ),
                    Linear( paramf( w["classification"]["w"] ),
                            paramf( w["classification"]["b"] ),
                            x->x )
                   )
end

function (c::XLNetClassifier)(x)
    #Size of x is 2 x BS
    token_ids = x[:,1,:]
    seg_ids = x[:,2,:]
    attn_mask = x[:,3,:]

    x = c.model( token_ids , seg_ids, attn_mask )
    #Note: getindex! doesn't bacprob properly for 3 dimensional arrays.
    y = permutedims( x, [2,1,3] )
    y = reshape( y,:,size(x,3) )
    y = y[end-size(x,2)+1:end, : ]
    y = tanh.(y)

    y = permutedims(y, [2,1] )
    y = c.projection(y)
    y = c.classification(y)
    y
end

(c::XLNetClassifier)(x,y) = nll( c(x),y )

include("weight_manager.jl")
include("preprocessing.jl")

end
