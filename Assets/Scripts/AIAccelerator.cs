using System;
using UnityEngine;
using Unity.Sentis;

public class AIAccelerator : MonoBehaviour {
    [SerializeField] private Simulation simulation;
    [SerializeField] private ModelAsset accelerationModel;
    [SerializeField] private bool operateOnToneMapped;

    public RenderTexture HDROutputTexture { get; private set; }
    public RenderTexture ToneMappedOutputTexture { get; private set; }

    Worker aiWorker;
    Tensor<float> sourceTensor;


    void Start() {
        if(simulation) {
            simulation.OnStep += Simulation_OnStep;
            HDROutputTexture = new RenderTexture(simulation.TextureResolution, simulation.TextureResolution, 0, RenderTextureFormat.ARGBFloat);
            HDROutputTexture.Create();
            ToneMappedOutputTexture = new RenderTexture(simulation.TextureResolution, simulation.TextureResolution, 0, RenderTextureFormat.ARGB32);
            ToneMappedOutputTexture.Create();
        }

        var model = ModelLoader.Load(accelerationModel);
        aiWorker = new Worker(model, BackendType.GPUCompute);

    }

    void OnDisable() {
        if(simulation) {
            simulation.OnStep -= Simulation_OnStep;
        }

        if(aiWorker != null) {
            aiWorker.Dispose();
            aiWorker = null;
        }

        if(sourceTensor != null) {
            sourceTensor.Dispose();
            sourceTensor = null;
        }

        if(HDROutputTexture != null) {
            DestroyImmediate(HDROutputTexture);
            HDROutputTexture = null;
        }

        if(ToneMappedOutputTexture) {
            DestroyImmediate(ToneMappedOutputTexture);
            ToneMappedOutputTexture = null;
        }
    }

    void Simulation_OnStep(int frameCount) {
        Tensor<float> outputTensor = null;

        if(operateOnToneMapped) {
            // Push output texture to input tensor
            sourceTensor = TextureConverter.ToTensor(simulation.SimulationOutputToneMapped);
            
            // Push input tensor through model
            aiWorker.Schedule(sourceTensor);
            outputTensor = aiWorker.PeekOutput() as Tensor<float>;

            // Push output tensor to final texture
            TextureConverter.RenderToTexture(outputTensor, ToneMappedOutputTexture);
        } else {
            // Push output texture to input tensor
            sourceTensor = TextureConverter.ToTensor(simulation.SimulationOutputHDR);
            
            // Push input tensor through model
            aiWorker.Schedule(sourceTensor);
            outputTensor = aiWorker.PeekOutput() as Tensor<float>;

            // Push output tensor to final texture
            TextureConverter.RenderToTexture(outputTensor, HDROutputTexture);
        }

        sourceTensor.Dispose();
        outputTensor.Dispose();
    }
}