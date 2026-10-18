// A4: the see-through blocks the automatic scan draws around objects (docs/auto-scan.md). One colour with its alpha,
// drawn after the opaque scene, both faces, without writing depth. In a Resources folder, so the Android build always
// includes it (a shader only found by name can be stripped). Single-pass instanced and multiview stereo: both eyes.
// Alpha is blended separately (One, OneMinusSrcAlpha) so the eye buffer's alpha stays the block's own coverage: the
// headset composites passthrough by that alpha, and SrcAlpha there would square it, leaving blocks nearly invisible.
Shader "SecondEyes/SeeThrough"
{
    Properties
    {
        _Color ("Colour", Color) = (0.25, 0.7, 1.0, 0.2)
    }
    SubShader
    {
        Tags { "Queue" = "Transparent" "RenderType" = "Transparent" "IgnoreProjector" = "True" }
        Pass
        {
            Blend SrcAlpha OneMinusSrcAlpha, One OneMinusSrcAlpha
            ZWrite Off
            Cull Off

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            #include "UnityCG.cginc"

            fixed4 _Color;

            struct appdata
            {
                float4 vertex : POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            v2f vert(appdata v)
            {
                v2f o;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_OUTPUT(v2f, o);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
                o.pos = UnityObjectToClipPos(v.vertex);
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                return _Color;
            }
            ENDCG
        }
    }
}
