Shader "RT/DiagnosticAShader"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _filter ("Filter", Color) = (1,1,1,1) 
        _lod ("Level of Detail", Integer) = 0
        _mapping ("Value Mapping", Range(-1,1)) = 0
        _low ("Lowerbound", Float) = 0
        _high ("Upperbound", Float) = 1
        _showAlpha ("Show Alpha", Float) = 0 
    }
    SubShader
    {
        Tags { "RenderType"="Transparent" }
        LOD 100

        Pass
        {
            Blend SrcAlpha OneMinusSrcAlpha
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                UNITY_FOG_COORDS(1)
                float4 vertex : SV_POSITION;
            };

            Texture2D<float4> _MainTex;
            SamplerState sampler_point_clamp;
            //sampler2D _MainTex;
            float4 _MainTex_ST;
            float _lod;
            float4 _filter;
            float _mapping;
            float _low;
            float _high;
            float _showAlpha;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                float4 sample = _MainTex.SampleLevel(sampler_point_clamp, i.uv, _lod);

                if(_showAlpha) {
                    sample.rgb = sample.aaa;
                }

                sample = (sample - _low) / (_high - _low);

                return pow(float4(sample.rgb, 1) * _filter, pow(10, -_mapping));
            }
            ENDCG
        }
    }
}
