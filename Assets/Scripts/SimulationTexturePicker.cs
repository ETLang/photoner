using UnityEngine;

[RequireComponent(typeof(Renderer))]
public class SimulationTexturePicker : MonoBehaviour {
    public enum TextureType {
        ToneMapped,
        HDR,
        AI_ToneMapped,
        AI_HDR,
        Albedo,
        Transmissibility,
        NormalSlope
    }

    [SerializeField] private Simulation simulation;
    [SerializeField] private AIAccelerator aiAccelerator;
    [SerializeField] private TextureType type = TextureType.ToneMapped;

    void OnDisable()
    {
        GetComponent<Renderer>().material.SetTexture("_MainTex", null);    
    }

    void LateUpdate()
    {
        if(!simulation) return;

        var renderer = GetComponent<Renderer>();
        Texture value = null;

        switch(type) {
        case TextureType.ToneMapped:
            value = simulation?.SimulationOutputToneMapped;
            break;
        case TextureType.HDR:
            value = simulation?.SimulationOutputHDR;
            break;
        case TextureType.AI_ToneMapped:
            value = aiAccelerator?.ToneMappedOutputTexture;
            break;
        case TextureType.AI_HDR:
            value = aiAccelerator?.HDROutputTexture;
            break;
        case TextureType.Albedo:
            value = simulation?.GBufferAlbedo;
            break;
        case TextureType.Transmissibility:
            value = simulation?.GBufferTransmissibility;
            break;
        case TextureType.NormalSlope:
            value = simulation?.GBufferNormalSlope;
            break;
        }

        if(value != null) {
            renderer.material.SetTexture("_MainTex", value);
        }
    }
}