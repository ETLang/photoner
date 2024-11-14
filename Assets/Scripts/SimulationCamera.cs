using System;
using UnityEngine;
using UnityEngine.Rendering;

[RequireComponent(typeof(Camera))]
public class SimulationCamera : MonoBehaviour {

    public ComputeShader Shader { get; set; }

    public RenderTexture GBufferAlbedo { get; set; }
    public RenderTexture GBufferTransmissibility { get; set; }
    public RenderTexture GBufferNormalSlope { get; set; }
    public RenderTexture GBufferQuadTreeLeaves { get; set; }
    public float VarianceEpsilon {
        get => _varianceEpsilon;
        set {
            if(_varianceEpsilon != value) {
                if(_postRenderCommands != null) {
                    _postRenderCommands.Dispose();
                    _postRenderCommands = null;
                }
            }
            _varianceEpsilon = value;
        }
    }
    private float _varianceEpsilon;

    public Texture2D TestTexture;

    public Action UpdateSimulation { get; set; }

    private CommandBuffer _postRenderCommands;

    void OnPreRender() {

        var gBuffer = new RenderBuffer[]
        {
            GBufferAlbedo.colorBuffer,
            GBufferTransmissibility.colorBuffer,
            GBufferNormalSlope.colorBuffer
        };

        RenderTexture rt = RenderTexture.active;
        RenderTexture.active = GBufferAlbedo;
        GL.Clear(false, true, new Color(0,0,0,1));
        RenderTexture.active = GBufferTransmissibility;
        GL.Clear(false, true, new Color(1,1,0,1));
        RenderTexture.active = GBufferNormalSlope;
        GL.Clear(false, true, new Color(0,0,0,0));
        RenderTexture.active = GBufferQuadTreeLeaves;
        GL.Clear(false, true, new Color(0,0,0,0));
        RenderTexture.active = rt;
        GetComponent<Camera>().SetTargetBuffers(gBuffer, GBufferAlbedo.depthBuffer);
    }

    void OnPostRender() {
        if(_postRenderCommands == null) {
           _postRenderCommands = new CommandBuffer();

            var generateGBufferMipsKernel = Shader.FindKernel("GenerateGBufferMips");
            int mipSize = GBufferTransmissibility.width;

            _postRenderCommands.SetComputeVectorParam(Shader, 
                "g_target_size", new Vector2(GBufferAlbedo.width, GBufferAlbedo.height));
            _postRenderCommands.SetComputeIntParam(Shader,
                "g_lowest_lod", (int)(GBufferAlbedo.mipmapCount - 3));

            for(int i = 1;i < GBufferTransmissibility.mipmapCount;i++) {
                mipSize /= 2;
                _postRenderCommands.SetComputeTextureParam(Shader, generateGBufferMipsKernel, 
                    "g_destMipLevelAlbedo", GBufferAlbedo, i);
                _postRenderCommands.SetComputeTextureParam(Shader, generateGBufferMipsKernel,
                    "g_sourceMipLevelAlbedo", GBufferAlbedo, i-1);
                _postRenderCommands.SetComputeTextureParam(Shader, generateGBufferMipsKernel,
                    "g_destMipLevelTransmissibility", GBufferTransmissibility, i);
                _postRenderCommands.SetComputeTextureParam(Shader, generateGBufferMipsKernel,
                    "g_sourceMipLevelTransmissibility", GBufferTransmissibility, i-1);
                _postRenderCommands.SetComputeTextureParam(Shader, generateGBufferMipsKernel,
                    "g_destMipLevelNormalSlope", GBufferNormalSlope, i);
                _postRenderCommands.SetComputeTextureParam(Shader, generateGBufferMipsKernel,   
                    "g_sourceMipLevelNormalSlope", GBufferNormalSlope, i-1);
                _postRenderCommands.DispatchCompute(Shader, generateGBufferMipsKernel,
                    Math.Max(1, mipSize / 8), Math.Max(1, mipSize / 8), 1);
            }

            mipSize = GBufferTransmissibility.width;
            var computeGBufferVarianceKernel = Shader.FindKernel("ComputeGBufferVariance");
            var eps = VarianceEpsilon;
            for(int i = 1;i < GBufferTransmissibility.mipmapCount;i++) {
                mipSize /= 2;
                eps /= 2.0f;
                _postRenderCommands.SetComputeFloatParam(Shader, 
                    "g_TransmissibilityVariationEpsilon", eps);
                _postRenderCommands.SetComputeTextureParam(Shader, computeGBufferVarianceKernel,
                    "g_sourceMipLevelTransmissibility", GBufferTransmissibility, i);
                _postRenderCommands.DispatchCompute(Shader, computeGBufferVarianceKernel,
                    Math.Max(1, mipSize / 8), Math.Max(1, mipSize / 8), 1);
            }

            var generateQuadTreeKernel = Shader.FindKernel("GenerateGBufferQuadTree");
            _postRenderCommands.SetComputeTextureParam(Shader, generateQuadTreeKernel,
                "g_transmissibility", GBufferTransmissibility);
            _postRenderCommands.SetComputeTextureParam(Shader, generateQuadTreeKernel,
                "g_destQuadTreeLeaves", GBufferQuadTreeLeaves, 0);
            _postRenderCommands.DispatchCompute(Shader, generateQuadTreeKernel,
                Math.Max(1, GBufferQuadTreeLeaves.width / 8), Math.Max(1, GBufferQuadTreeLeaves.height / 8), 1);

        }

        Graphics.ExecuteCommandBuffer(_postRenderCommands);

        UpdateSimulation();
    }    
}