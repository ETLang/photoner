Shader "RT/Object"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _Color ("Color", Color) = (1,1,1,1) 
    }
    SubShader
    {
        Tags { "RenderType"="Transparent" }
        LOD 100

        Pass
        {
            Blend 0 SrcAlpha OneMinusSrcAlpha
            Blend 1 Zero SrcColor
            // [OPTIMIZED] Blend 1 SrcAlpha SrcColor

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            struct gbuffer_output
            {
                float4 albedo : SV_Target0;
                float4 transmissibility: SV_Target1;
                float4 normal : SV_Target2;
            };

            sampler2D _MainTex;
            float4x4 _worldToSimulationMatrix;
            float4 _MainTex_ST;
            float4 _Color;
            float _substrateDensity;

            sampler2D _NormalTex;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                return o;
            }

            gbuffer_output frag (v2f i)
            {
                gbuffer_output output;

                // sample the texture
                float4 c = tex2D(_MainTex, i.uv);
                float4 n = tex2D(_NormalTex, i.uv);



                // T = 1 / (1 + S)
                // S = 1/T - 1
                // Src*Dest
                // Src = 1-_substrateDensity*t.a
                // Dest = 1 / (1+Sd)
                // Ts=1-_substrateDensity*t.a
                // Ss = 1/Ts
                // Sfinal = 1/(Src*Dest)-1 = 1/(Ts / (1+Sd))-1
                //        = (1+Sd)/Ts-1 = 1/Ts + Sd/Ts - 1 = Sd * 1/Ts + 1/Ts - 1
                // 1/Ts - 1 = 1/Ts * tAlpha;
                // tAlpha = (1/Ts - 1) / (1/Ts) = Ts * (1/Ts - 1) = 1 - 1/Ts
                // 1/Ts - 1 = 1/Ts * (1 - 1/Ts)

                output.albedo = float4(c.rgb * _Color.rgb,c.a * _Color.a);
                float t = 1-_substrateDensity*c.a/sqrt(_ScreenParams.x*_ScreenParams.y)*100;
                float rT = 1/t;

                //output.transmissibility = float4(rT,rT,0,1-rT);
                output.transmissibility = float4(t,t,0,1);
                output.normal = float4(0,1,0,0);

                // output.albedo = float4(t.rgb * _Color.rgb, t.a * _Color.a);
                // output.transmissibility = float4(1 - (_substrateDensity * t.a * _Color.a),1,0,1);
                // output.transmissibility.y = output.transmissibility.x;
                // output.normal = float4(n.xy,0,0);

                return output;
            }
            ENDCG
        }
    }
}
