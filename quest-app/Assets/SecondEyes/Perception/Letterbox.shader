// Letterboxes a frame into the detector's square input (A1.10c, D84): the image keeps its aspect ratio at the top-left
// corner, the rest is the pad value, as YOLOX's preproc does. Output values are the source's stored 8-bit values:
// when the project is in linear colour space and the source is an sRGB texture, sampling linearizes and _EncodeSRGB
// encodes back. The target is a linear (non-sRGB) render texture, so nothing converts on write.
Shader "Hidden/SecondEyes/Letterbox"
{
    Properties
    {
        _MainTex ("Source", 2D) = "white" {}
    }
    SubShader
    {
        Cull Off ZWrite Off ZTest Always
        Pass
        {
            CGPROGRAM
            #pragma vertex vert_img
            #pragma fragment frag
            #pragma target 3.0
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            float4 _Region;      // x: image width / canvas, y: image height / canvas, z: pad value in [0, 1]
            float _EncodeSRGB;   // 1: encode linear samples back to sRGB values
            float _FlipSource;   // 1: the source's rows run bottom to top

            float4 frag (v2f_img i) : SV_Target
            {
                float fromTop = 1.0 - i.uv.y;
                if (i.uv.x >= _Region.x || fromTop >= _Region.y)
                    return float4(_Region.z, _Region.z, _Region.z, 1.0);
                float2 src = float2(i.uv.x / _Region.x, 1.0 - fromTop / _Region.y);
                if (_FlipSource > 0.5)
                    src.y = 1.0 - src.y;
                float4 c = tex2D(_MainTex, src);
                if (_EncodeSRGB > 0.5)
                {
                    c.r = LinearToGammaSpaceExact(c.r);
                    c.g = LinearToGammaSpaceExact(c.g);
                    c.b = LinearToGammaSpaceExact(c.b);
                }
                return float4(c.rgb, 1.0);
            }
            ENDCG
        }
    }
}
