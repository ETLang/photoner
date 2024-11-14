Shader "Hidden/Gauss4Sample"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _MainTexLOD ("Texture LOD", Float) = 0
    }
    SubShader
    {
        // No culling or depth
        Cull Off ZWrite Off ZTest Always
        Blend Off

        Pass
        {
            HLSLPROGRAM
            #pragma vertex vert_common
            #pragma fragment frag

            #include "RayTracing2DCommon.cginc"

            float4 frag(v2f_common i) : SV_Target
            {
                float2 sampleDelta = (1 - _ScreenParams.zw) / 2;
                return (
                    SAMPLE_LOD(_MainTex, i.uv + float2(1, 1) * sampleDelta, _MainTexLOD) +
                    SAMPLE_LOD(_MainTex, i.uv + float2(-1, 1) * sampleDelta, _MainTexLOD) +
                    SAMPLE_LOD(_MainTex, i.uv + float2(1, -1) * sampleDelta, _MainTexLOD) +
                    SAMPLE_LOD(_MainTex, i.uv + float2(-1, -1) * sampleDelta, _MainTexLOD)
                    ) / 4;
            }
            ENDHLSL
        }
    }
}
