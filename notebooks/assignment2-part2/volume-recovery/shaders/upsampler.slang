import texture;

public float4 sampling_fwd(
    uint2 tid,
    uint2 canvasSize,
    StructuredBuffer<float4> sampleTex,
    uint2 texSize)
{
    // TODO: Student implementation starts here.
    return sampleBilinear(DiffTexture2D<float, 4>(sampleTex, texSize), float2(tid) / float2(canvasSize));
    // TODO: Student implementation ends here.
}

public void sampling_bwd(
    uint2 tid,
    uint2 canvasSize,
    StructuredBuffer<float4> sampleTex,
    RWStructuredBuffer<Atomic<float>[4]> sampleTexGrad,
    uint2 texSize,
    float4 dOut)
{
    // TODO: Student implementation starts here.
    DifferentialPtrPair<DiffTexture2D<float, 4>> diffSampleTexPair = 
        DifferentialPtrPair<DiffTexture2D<float, 4>>(
            DiffTexture2D<float, 4>(sampleTex, texSize), 
            DiffTexture2D_Differential<float, 4>(sampleTexGrad)
    );    
    float2 uv = float2(tid) / float2(canvasSize);
    bwd_diff(sampleBilinear)(diffSampleTexPair, uv, dOut);

    // TODO: Student implementation ends here.
}
