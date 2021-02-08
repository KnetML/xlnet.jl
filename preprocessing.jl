#=
This function converts given text sample to token ids, adjust its length based on the sequance length and
adds special tokens at the end of the sample.
=#

function prepare_sample( text, seq_len, sp )
  input_ids = sp.encode_as_ids( text )
  input_length = length(input_ids)

  if( input_length > seq_len - 2 ); input_length = seq_len - 2; end
  input_ids = input_ids[1:input_length]

  push!(input_ids, specaialTokens["<SEP>"] )
  push!(input_ids, specaialTokens["<CLS>"] )

  attn_mask = zeros(Int32, seq_len)
  padded_input_ids = zeros(Int32, seq_len)

  attn_mask[ end-input_length-1 : end ] .= 1
  attn_mask = 1 .- attn_mask

  padded_input_ids[ end-input_length-1 : end ] = input_ids

  seq_ids = zeros(Int32,seq_len)
  seq_ids[end] = 2
  seq_ids[1:end-input_length] .= 4


  (padded_input_ids, seq_ids, attn_mask)
end
