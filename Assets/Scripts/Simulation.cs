using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Mathematics;
using UnityEngine.Rendering;
using Unity.Collections;
using System.Linq;


public delegate void SimulationStepEvent(int frameCount);
public delegate void SimulationConvergedEvent();

[Serializable]
public struct SimulationProfile {
    public int frameLimit;
    public int resolution;
    public int threadCount;
    public int photonsPerThread;
    public int photonBounces;
    public int energyUnit;
    public float transmissibilityVariationEps;
    public float outscatterCoefficient;
}

public class Simulation : MonoBehaviour
{
    private Camera real_contentCamera;
    [SerializeField] private LayerMask rayTracedLayers;

    [SerializeField] private int frameLimit = -1;
    [SerializeField] private int textureResolution = 256;

    [SerializeField] private int threadCount = 4096;
    [SerializeField] private int photonsPerThread = 4096;
    [SerializeField] private int photonBounces = -1;

    [SerializeField] private int energyUnit = 100000;
    [SerializeField] private float transmissibilityVariationEpsilon = 1e-3f;
    [SerializeField, Range(0,0.5f)] private float outscatterCoefficient = 0.01f;

    private ComputeShader _computeShader;
    private ComputeBuffer _randomBuffer;
    private ComputeBuffer _measureConvergenceResultBuffer;

    private RenderTexture[] _renderTexture = new RenderTexture[2];
    private int _currentRenderTextureIndex = 0;
    private RenderBuffer[] _gBuffer;
    private Texture _mieScatteringLUT;
    private int[] _kernelsHandles;

    private Renderer _renderer;

    [Header("Convergence Information")]
    [SerializeField] private int framesSinceClear = 0;
    [SerializeField, ReadOnly] private float convergenceProgress = 100000;

    private bool awaitingConvergenceResult = false;
    [SerializeField, ReadOnly] public bool hasConverged = false;

    private uint[] convergenceResultResetData = new uint[] {0, 0, 0, 0};
    private SortedDictionary<float,uint> performanceCounter = new SortedDictionary<float, uint>();

    public uint TraversalsPerSecond { get; private set; }

    public RenderTexture GBufferAlbedo { get; private set; }
    public RenderTexture GBufferTransmissibility { get; private set; }
    public RenderTexture GBufferNormalSlope { get; private set; }

    public RenderTexture SimulationOutputRaw { get; private set; }
    public RenderTexture SimulationOutputHDR { get; private set; }
    public RenderTexture SimulationOutputToneMapped { get; private set; }

    public float ConvergenceStartTime { get; private set; }
    public float Convergence => convergenceProgress;
    public int TextureResolution => textureResolution;

    public event SimulationStepEvent OnStep;
    public event SimulationConvergedEvent OnConverged;

    public void LoadProfile(SimulationProfile profile) {
        frameLimit = profile.frameLimit;
        textureResolution = profile.resolution;
        threadCount = profile.threadCount;
        photonsPerThread = profile.photonsPerThread;
        energyUnit = profile.energyUnit;
        transmissibilityVariationEpsilon = profile.transmissibilityVariationEps;
        outscatterCoefficient = profile.outscatterCoefficient;
        photonBounces = profile.photonBounces;
        hasConverged = false;
        framesSinceClear = 0;
    }

    private void Start()
    {
        _computeShader = (ComputeShader)Resources.Load("Test_Compute");
        Training.shader = _computeShader;

        //GET RENDERER COMPONENT REFERENCE
        TryGetComponent(out _renderer);

        uint4[] seeds = new uint4[threadCount];

        for(int i = 0;i < seeds.Length;i++) {
            seeds[i].x = (uint)(UnityEngine.Random.value * 1000000);
            seeds[i].y = (uint)(UnityEngine.Random.value * 1000000);
            seeds[i].z = (uint)(UnityEngine.Random.value * 1000000);
            seeds[i].w = (uint)(UnityEngine.Random.value * 1000000);
        }

        _randomBuffer = new ComputeBuffer(threadCount, 16, ComputeBufferType.Structured, ComputeBufferMode.Immutable);
        _randomBuffer.SetData(seeds);
        
        _measureConvergenceResultBuffer = new ComputeBuffer(1, 16, ComputeBufferType.Structured, ComputeBufferMode.Immutable);
        _measureConvergenceResultBuffer.SetData(convergenceResultResetData);

        //CREATE NEW RENDER TEXTURE TO RENDER DATA TO
        
        for(int i = 0;i < _renderTexture.Length;i++) {
            _renderTexture[i] = new RenderTexture(textureResolution, textureResolution, 0, RenderTextureFormat.DefaultHDR)
            {
                enableRandomWrite = true
            };
            _renderTexture[i].Create();
        }

        SimulationOutputRaw = new RenderTexture(textureResolution * 3, textureResolution, 0, RenderTextureFormat.RInt)
        {
            enableRandomWrite = true
        };
        SimulationOutputRaw.Create();

        SimulationOutputHDR = new RenderTexture(textureResolution, textureResolution, 0, RenderTextureFormat.ARGBFloat)
        {
            enableRandomWrite = true
        };
        SimulationOutputHDR.Create();

        GBufferAlbedo = new RenderTexture(textureResolution, textureResolution, 0, RenderTextureFormat.ARGBFloat)
        {
            enableRandomWrite = true,
            useMipMap = true,
            autoGenerateMips = false
        };
        GBufferAlbedo.Create();
        GBufferAlbedo.GenerateMips();

        GBufferTransmissibility = new RenderTexture(textureResolution, textureResolution, 0, RenderTextureFormat.ARGBFloat)
        {
            enableRandomWrite = true,
            useMipMap = true,
            autoGenerateMips = false
        };
        GBufferTransmissibility.Create();
        GBufferTransmissibility.GenerateMips();

        GBufferNormalSlope = new RenderTexture(textureResolution, textureResolution, 0, RenderTextureFormat.ARGBFloat)
        {
            enableRandomWrite = true,
            useMipMap = true,
            autoGenerateMips = false
        };
        GBufferNormalSlope.Create();
        GBufferNormalSlope.GenerateMips();

        _gBuffer = new RenderBuffer[]
        {
            GBufferAlbedo.colorBuffer,
            GBufferTransmissibility.colorBuffer,
            GBufferNormalSlope.colorBuffer
        };


        _mieScatteringLUT = LUT.CreateMieScatteringLUT();

        real_contentCamera = new GameObject("__Simulation_Camera", typeof(Camera)).GetComponent<Camera>();
        real_contentCamera.transform.parent = transform;
        real_contentCamera.transform.localScale = new Vector3(1,1,1);
        real_contentCamera.transform.localRotation = Quaternion.identity;
        real_contentCamera.transform.localPosition = Vector3.zero;
        real_contentCamera.orthographic = true;
        real_contentCamera.orthographicSize = transform.localScale.x / 2;
        real_contentCamera.nearClipPlane = -1;
        real_contentCamera.farClipPlane = 1;
        real_contentCamera.cullingMask = rayTracedLayers.value;
        real_contentCamera.clearFlags = CameraClearFlags.Nothing;
        real_contentCamera.allowHDR = false;
        real_contentCamera.allowMSAA = false;
        real_contentCamera.useOcclusionCulling = false;
        real_contentCamera.gameObject.SetActive(false);
    }

    private void OnDisable()
    {
        DestroyImmediate(real_contentCamera);
        real_contentCamera = null;

        for(int i = 0;i < _renderTexture.Length;i++) {
            if (_renderTexture[i] != null)
            {
                DestroyImmediate(_renderTexture[i]);
                _renderTexture[i] = null;
            }
        }

        DestroyImmediate(SimulationOutputRaw);
        SimulationOutputRaw = null;

        DestroyImmediate(SimulationOutputHDR);
        SimulationOutputHDR = null;

        DestroyImmediate(GBufferAlbedo);
        GBufferAlbedo = null;

        DestroyImmediate(GBufferTransmissibility);
        GBufferTransmissibility = null;

        DestroyImmediate(GBufferNormalSlope);
        GBufferNormalSlope = null;

        _randomBuffer.Release();
        _randomBuffer = null;
        _measureConvergenceResultBuffer.Release();
        _measureConvergenceResultBuffer = null;
    }

    Matrix4x4 _previousSimulationMatrix;
    HashSet<RTLightSource> _previousLightSources = new HashSet<RTLightSource>();
    HashSet<RTObject> _previousObjects = new HashSet<RTObject>();
    float _previousOutscatterCoefficient;
    int _sceneId;

    void Update()
    {
        var worldToPresentationSpace = transform.worldToLocalMatrix;
        var presentationToTargetSpace = Matrix4x4.Translate(new Vector3(0.5f, 0.5f, 0));
        var worldToTargetSpace = presentationToTargetSpace * worldToPresentationSpace;
        double photonEnergy = (double)uint.MaxValue / threadCount;
        var allLights = GameObject.FindObjectsByType<RTLightSource>(FindObjectsSortMode.None);
        var allObjects = GameObject.FindObjectsByType<RTObject>(FindObjectsSortMode.None);
        float energyNormPerFrame = (float)photonsPerThread * (float)threadCount / (textureResolution * textureResolution);

        var now = Time.time;
        while(performanceCounter.Keys.Count != 0 && performanceCounter.Keys.First() < now - 1)
            performanceCounter.Remove(performanceCounter.Keys.First());
        
        uint total = 0;
        foreach(var value in performanceCounter.Values)
            total += value;
        uint bouncesThisFrame = 0;
        
        foreach(var bounces in allLights.Select(light => light.bounces))
            bouncesThisFrame += bounces;

        bouncesThisFrame *= (uint)photonsPerThread * (uint)threadCount;
        
        if(performanceCounter.TryGetValue(now, out var existing)) {
            performanceCounter[now] = existing + bouncesThisFrame;
        } else {
            performanceCounter[now] = bouncesThisFrame;
        }

        TraversalsPerSecond = total;

        // CHANGE DETECTION
        if( allLights.Length != _previousLightSources.Count ||
            !allLights.All(l => _previousLightSources.Contains(l)) ||
            allLights.Any(l => l.Changed) ||
            allObjects.Length != _previousObjects.Count ||
            !allObjects.All(o => _previousObjects.Contains(o)) ||
            allObjects.Any(o => o.Changed) ||
            _previousSimulationMatrix != worldToPresentationSpace ||
            _previousOutscatterCoefficient != outscatterCoefficient) {
            hasConverged = false;
            framesSinceClear = 0;

            _previousOutscatterCoefficient = outscatterCoefficient;
            _previousSimulationMatrix = worldToPresentationSpace;
            _previousLightSources.Clear();
            foreach(var light in allLights)
                _previousLightSources.Add(light);
            _previousObjects.Clear();
            foreach(var o in allObjects)
                _previousObjects.Add(o);
            _sceneId++;
        }

        if(hasConverged) return;

        // CLEAR TARGET
        if(framesSinceClear == 0) {
            awaitingConvergenceResult = false;

            RenderTexture rt = RenderTexture.active;
            RenderTexture.active = SimulationOutputRaw;
            GL.Clear(true, true, Color.clear);
            RenderTexture.active = rt;
            convergenceProgress = -1;
            ConvergenceStartTime = now;
        }

        framesSinceClear++;

        // G BUFFER PRODUCTION
        {
            RenderTexture rt = RenderTexture.active;
            RenderTexture.active = GBufferAlbedo;
            GL.Clear(false, true, new Color(0,0,0,1));
            RenderTexture.active = GBufferTransmissibility;
            GL.Clear(false, true, new Color(1,1,0,1));
            RenderTexture.active = GBufferNormalSlope;
            GL.Clear(false, true, new Color(0,0,0,0));
            RenderTexture.active = rt;

            real_contentCamera.SetTargetBuffers(_gBuffer, GBufferAlbedo.depthBuffer);
            real_contentCamera.Render();
        }

        // MIPMAP PRODUCTION
        var generateGBufferMipsKernel = _computeShader.FindKernel("GenerateGBufferMips");
        int mipSize = GBufferTransmissibility.width;
        for(int i = 1;i < GBufferTransmissibility.mipmapCount;i++) {
            mipSize /= 2;
            _computeShader.SetTexture(generateGBufferMipsKernel, "g_destMipLevelAlbedo", GBufferAlbedo, i);
            _computeShader.SetTexture(generateGBufferMipsKernel, "g_sourceMipLevelAlbedo", GBufferAlbedo, i-1);
            _computeShader.SetTexture(generateGBufferMipsKernel, "g_destMipLevelTransmissibility", GBufferTransmissibility, i);
            _computeShader.SetTexture(generateGBufferMipsKernel, "g_sourceMipLevelTransmissibility", GBufferTransmissibility, i-1);
            _computeShader.SetTexture(generateGBufferMipsKernel, "g_destMipLevelNormalSlope", GBufferNormalSlope, i);
            _computeShader.SetTexture(generateGBufferMipsKernel, "g_sourceMipLevelNormalSlope", GBufferNormalSlope, i-1);
            _computeShader.Dispatch(generateGBufferMipsKernel, Math.Max(1, mipSize / 8), Math.Max(1, mipSize / 8), 1);
        }

        // RAY TRACING SIMULATION
        _computeShader.SetVector("g_target_size", new Vector2(textureResolution, textureResolution));
        _computeShader.SetInt("g_time_ms", Time.frameCount);
        _computeShader.SetInt("g_photons_per_thread", photonsPerThread);
        _computeShader.SetFloat("g_energy_norm", framesSinceClear * energyNormPerFrame * energyUnit);
        _computeShader.SetMatrix("g_worldToTarget", Matrix4x4.identity);
        _computeShader.SetFloat("g_TransmissibilityVariationEpsilon", transmissibilityVariationEpsilon);
        _computeShader.SetInt("g_lowest_lod", (int)(GBufferTransmissibility.mipmapCount - 1));
        _computeShader.SetInt("g_4x4_lod", (int)(GBufferTransmissibility.mipmapCount - 3));
        _computeShader.SetFloat("g_lightEmissionOutscatter", 0);
        _computeShader.SetFloat("g_outscatterCoefficient", outscatterCoefficient);

        foreach(var light in allLights) {
            int simulateKernel = -1;
            var lightToTargetSpace = worldToTargetSpace * light.WorldTransform;

            switch(light) {
            case RTPointLight pt:
                simulateKernel = _computeShader.FindKernel("Simulate_PointLight");
                _computeShader.SetFloat("g_lightEmissionOutscatter", pt.emissionOutscatter);
                break;
            case RTSpotLight _:
                simulateKernel = _computeShader.FindKernel("Simulate_SpotLight");
                break;
            case RTLaserLight _:
                simulateKernel = _computeShader.FindKernel("Simulate_LaserLight");
                break;
            case RTAmbientLight _:
                simulateKernel = _computeShader.FindKernel("Simulate_AmbientLight");
                break;
            case RTFieldLight field:
                simulateKernel = _computeShader.FindKernel("Simulate_FieldLight");
                _computeShader.SetTexture(simulateKernel, "g_lightFieldTexture", field.lightTexture ? field.lightTexture : Texture2D.whiteTexture);
                _computeShader.SetFloat("g_lightEmissionOutscatter", field.emissionOutscatter);
                break;
            case RTDirectionalLight dir:
                simulateKernel = _computeShader.FindKernel("Simulate_DirectionalLight");
                _computeShader.SetVector("g_directionalLightDirection", lightToTargetSpace.MultiplyVector(new Vector3(0,-1,0)));
                break;
            }


            _computeShader.SetVector("g_lightEnergy", light.Energy * (float)photonEnergy);
            _computeShader.SetInt("g_bounces", photonBounces != -1 ? photonBounces : (int)light.bounces);
            _computeShader.SetMatrix("g_lightToTarget", lightToTargetSpace.transpose);
            _computeShader.SetBuffer(simulateKernel, "g_rand", _randomBuffer);
            _computeShader.SetTexture(simulateKernel, "g_photons", SimulationOutputRaw);
            _computeShader.SetTexture(simulateKernel, "g_albedo", GBufferAlbedo);
            _computeShader.SetTexture(simulateKernel, "g_transmissibility", GBufferTransmissibility);
            _computeShader.SetTexture(simulateKernel, "g_normalSlope", GBufferNormalSlope);
            _computeShader.SetTexture(simulateKernel, "g_mieScatteringLUT", _mieScatteringLUT);

            _computeShader.Dispatch(simulateKernel, threadCount / 64, 1, 1);
        }

        // HDR MAPPING
        var hdrKernel = _computeShader.FindKernel("ConvertToHDR");
        _computeShader.SetTexture(hdrKernel, "g_photons", SimulationOutputRaw);
        _computeShader.SetTexture(hdrKernel, "g_hdrResult", SimulationOutputHDR);
        _computeShader.Dispatch(hdrKernel, (textureResolution - 1) / 8 + 1, (textureResolution - 1) / 8 + 1, 1);

        // TONE MAPPING
        var toneMapKernel = _computeShader.FindKernel("ToneMap");
        _computeShader.SetTexture(toneMapKernel, "g_photons", SimulationOutputRaw);
        _computeShader.SetTexture(toneMapKernel, "g_result", _renderTexture[_currentRenderTextureIndex]);
        _computeShader.Dispatch(toneMapKernel, (textureResolution - 1) / 8 + 1, (textureResolution - 1) / 8 + 1, 1);

        SimulationOutputToneMapped = _renderTexture[_currentRenderTextureIndex];

        OnStep?.Invoke(framesSinceClear);
        
        // CONVERGENCE TESTING
        if(frameLimit != -1 && framesSinceClear >= frameLimit) {
            if(framesSinceClear > frameLimit) {
                Debug.LogError("Skipped a frame somehow...");
            }

            hasConverged = true;
            OnConverged?.Invoke();
        }

        int framesPerTest = bouncesThisFrame == 0 ? 0 : 10 * textureResolution * textureResolution / (int)bouncesThisFrame;
        framesPerTest = Math.Max(framesPerTest, 10);
        if(!awaitingConvergenceResult && framesPerTest != 0 && framesSinceClear % framesPerTest == 0) {
            awaitingConvergenceResult = true;

            _measureConvergenceResultBuffer.SetData(convergenceResultResetData);

            var measureConvergenceKernel = _computeShader.FindKernel("MeasureConvergence");
            _computeShader.SetTexture(measureConvergenceKernel, "g_result", _renderTexture[_currentRenderTextureIndex]);
            _computeShader.SetTexture(measureConvergenceKernel, "g_previousResult", _renderTexture[1-_currentRenderTextureIndex]);
            _computeShader.SetBuffer(measureConvergenceKernel, "g_convergenceResult", _measureConvergenceResultBuffer);
            _computeShader.Dispatch(measureConvergenceKernel, (textureResolution - 1) / 8 + 1, (textureResolution - 1) / 8 + 1, 1);
            _currentRenderTextureIndex = 1 - _currentRenderTextureIndex;

            int recentSceneId = _sceneId;
            AsyncGPUReadback.Request(_measureConvergenceResultBuffer, (r) =>
            {
                if(recentSceneId != _sceneId) return;

                awaitingConvergenceResult = false;
                if(!r.done || r.hasError) return;

                NativeArray<uint> feedback = r.GetData<uint>(0);

                var totalPixels = photonsPerThread * threadCount;
                double eps = totalPixels * 1e-8;

                float nextConvergenceProgress;
                if(convergenceProgress != -1) {
                    nextConvergenceProgress = convergenceProgress * 0.9f + 0.1f * Math.Max(0, (float)((double)feedback[0] / eps - 1));
                    hasConverged = nextConvergenceProgress < 1000 && (nextConvergenceProgress + 1e-8f > convergenceProgress);

                    if(hasConverged) {
                        OnConverged?.Invoke();
                    }
                } else {
                    nextConvergenceProgress = Math.Max(0, (float)((double)feedback[0] / eps - 1));
                }
                convergenceProgress = nextConvergenceProgress;
            });
        }
    }
}
