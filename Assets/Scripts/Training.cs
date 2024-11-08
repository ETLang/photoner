using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Mathematics;
using UnityEngine.Rendering;
using Unity.Collections;
using System.Linq;

using Random=System.Random;

public static class RandExt {
    public static float NextSingle(this Random rand) {
        return (float)rand.NextDouble();
    }

    public static float NextRange(this Random rand, float min, float max, float bias = 0) {
        return Mathf.Pow(rand.NextSingle(), Mathf.Pow(10, -bias)) * (max - min) + min;
    }

    public static bool NextBool(this Random rand, float weight = 0.5f) {
        return rand.NextSingle() < weight;
    }

    public static T NextWeightedOption<T>(this Random rand, Dictionary<T,float> weights) {
        var total = weights.Values.Sum();
        var val = rand.NextSingle() * total;

        foreach(var kvp in weights)
        {
            if(val <= kvp.Value)
                return kvp.Key;
            val -= kvp.Value;
        }
        return weights.Keys.Last();
    }

    public static Color NextLightColor(this Random rand) {
        return Color.HSVToRGB(rand.NextSingle(), Mathf.Sqrt(rand.NextSingle()), 1);
    }
}

[System.Serializable]
public class SubstrateElementTransform {
    public Vector2 position;
    public Vector2 scale;
    [Range(0, 360)] public float angle;

    public Matrix4x4 matrix => Matrix4x4.TRS((Vector3)position, Quaternion.AngleAxis(angle, new Vector3(0,0,1)), new Vector3(scale.x, scale.y, 1));
}

[System.Serializable]
public class SubstrateTrainingData {
    public int textureSize;
    [ReadOnly, HexInt(8)] public uint seed;
    public SubstrateElementTransform[] rects;
    public SubstrateElementTransform[] ellipses;
    public SubstrateElementTransform[] inverseRects;
    public SubstrateElementTransform[] inverseEllipses;
    [Range(1,256)] public float edgeBlur;
    [Range(-1,1)] public float sharpness;

    [Header("Noise")]
    public bool hasNoise;
    [Range(0,9)] public int minFrequency;
    [Range(0,9)] public int maxFrequency;
    [Range(0,1)] public float floor;
    [Range(0,1)] public float ceiling;

    [Header("Color")]
    public Color colorA;
    public Color colorB;
    [Range(0,1)] public float densityA;
    [Range(0,1)] public float densityB;
    [Range(0,360)] public float gradientAngle;
    [Range(0.1f, 1.4f)] public float gradientLength;
}

public static class Training
{
    public static ComputeShader shader { get; set; }

    public static Texture GenerateOneSubstrate(int textureSize, int seed) {
        System.Random rand = new System.Random(seed);

        var target = new RenderTexture[2];
        int destTarget = 0;
        for(int i = 0;i < target.Length;i++) {
            target[i] = new RenderTexture(textureSize, textureSize, 0, RenderTextureFormat.ARGBFloat)
            {
                enableRandomWrite = true
            };
            target[i].Create();
        }

        var rectCount = rand.Next(4);
        var ellipseCount = rand.Next(4);
        var invRectCount = rand.Next(3);
        var invEllipseCount = rand.Next(3);

        Matrix4x4[] shapeTransforms = new Matrix4x4[16];

        {
            int matrixIndex = 0;
            for(int i = 0;i < rectCount;i++) {
                shapeTransforms[matrixIndex] = Matrix4x4.TRS(
                    new Vector3(rand.NextRange(-0.9f, 0.9f), rand.NextRange(-0.9f, 0.9f), 0),
                    Quaternion.AngleAxis(rand.NextRange(0, 360.0f), new Vector3(0,0,1)),
                    new Vector3(rand.NextRange(0.1f, 0.5f), rand.NextRange(0.1f, 0.5f), 1)
                ).inverse.transpose;
                matrixIndex++;
            }
            for(int i = 0;i < ellipseCount;i++) {
                shapeTransforms[matrixIndex] = Matrix4x4.TRS(
                    new Vector3(rand.NextRange(-0.9f, 0.9f), rand.NextRange(-0.9f, 0.9f), 0),
                    Quaternion.AngleAxis(rand.NextRange(0, 360.0f), new Vector3(0,0,1)),
                    new Vector3(rand.NextRange(0.1f, 0.5f), rand.NextRange(0.1f, 0.5f), 1)
                ).inverse.transpose;
                matrixIndex++;
            }
            for(int i = 0;i < invRectCount;i++) {
                shapeTransforms[matrixIndex] = Matrix4x4.TRS(
                    new Vector3(rand.NextRange(-0.7f, 0.7f), rand.NextRange(-0.7f, 0.7f), 0),
                    Quaternion.AngleAxis(rand.NextRange(0, 360.0f), new Vector3(0,0,1)),
                    new Vector3(rand.NextRange(0.1f, 0.3f), rand.NextRange(0.1f, 0.3f), 1)
                ).inverse.transpose;
                matrixIndex++;
            }
            for(int i = 0;i < invEllipseCount;i++) {
                shapeTransforms[matrixIndex] = Matrix4x4.TRS(
                    new Vector3(rand.NextRange(-0.7f, 0.7f), rand.NextRange(-0.7f, 0.7f), 0),
                    Quaternion.AngleAxis(rand.NextRange(0, 360.0f), new Vector3(0,0,1)),
                    new Vector3(rand.NextRange(0.1f, 0.3f), rand.NextRange(0.1f, 0.3f), 1)
                ).inverse.transpose;
                matrixIndex++;
            }
        }


        float edgeBlur = rand.NextRange(1.0f, 256.0f);
        float hardness = Mathf.Pow(10,rand.NextRange(-1,1));
        bool hasNoise = rand.NextBool();
        int minFrequency = rand.Next(6);
        int maxFrequency = minFrequency + rand.Next(5);

        minFrequency = 1 << minFrequency;
        maxFrequency = 1 << maxFrequency;

        var noiseFloor = rand.NextRange(0, 0.4f);
        var noiseCeiling = rand.NextRange(0.6f, 1);

        var colorA = Color.HSVToRGB(rand.NextSingle(), rand.NextSingle(), 1);
        var colorB = Color.HSVToRGB(rand.NextSingle(), rand.NextSingle(), 1);
        var intensityA = Mathf.Min(Mathf.Pow(10, rand.NextRange(-4, 0)), 0.9f);
        var intensityB = Mathf.Min(Mathf.Pow(10, rand.NextRange(-4, 0)), 0.9f);
        
        var gradientAngle = rand.NextRange(0, 2*Mathf.PI);
        var gradientLength = rand.NextRange(0.1f, 1.4f);

        Vector3 gradient = new Vector3(
            Mathf.Cos(gradientAngle) * gradientLength, 
            Mathf.Sin(gradientAngle) * gradientLength, 
            -gradientLength / 2.0f);

        bool hasGradient = rand.NextBool();

        if(!hasGradient) {
            colorB = colorA;
            intensityB = intensityA;
        }

        shader.SetVector("g_trainingSubstrate_size", new Vector2(textureSize, textureSize));
        shader.SetVector("g_trainingSubstrate_ShapeCounts", new Vector4(rectCount, ellipseCount, invRectCount, invEllipseCount));
        shader.SetMatrixArray("g_trainingSubstrate_ShapeTransforms", shapeTransforms);
        shader.SetFloat("g_trainingSubstrate_edgeBlur", edgeBlur);
        shader.SetFloat("g_trainingSubstrate_hardness", hardness);
        shader.SetVector("g_trainingSubstrate_noiseFrequencyMinMax", new Vector2(minFrequency, maxFrequency));
        shader.SetVector("g_trainingSubstrate_noiseClipValues", new Vector2(noiseFloor, noiseCeiling));
        shader.SetVector("g_trainingSubstrate_gradientDirection", gradient);
        shader.SetVector("g_trainingSubstrate_gradientIntensity", new Vector2(intensityA, intensityB));
        shader.SetVector("g_trainingSubstrate_gradientColorA", colorA);
        shader.SetVector("g_trainingSubstrate_gradientColorB", colorB);

        int kernel;
        // Make shapes (rects and ellipses)
        kernel = shader.FindKernel("GenerateTrainingSubstrate_MakeShapes");
        shader.SetTexture(kernel, "g_trainingSubstrate_textureRef", target[1-destTarget]);
        shader.SetTexture(kernel, "g_trainingSubstrate_textureDest", target[destTarget]);
        shader.Dispatch(kernel, Math.Max(1, textureSize / 8), Math.Max(1, textureSize / 8), 1);

        // Blur edge
        kernel = shader.FindKernel("GenerateTrainingSubstrate_EdgeBlur_JFA");
        for(int i = 1;i < textureSize;i *= 2) {
            shader.SetInt("g_trainingSubstrate_edgeBlurStage", i);
            destTarget = 1 - destTarget;
            shader.SetTexture(kernel, "g_trainingSubstrate_textureRef", target[1-destTarget]);
            shader.SetTexture(kernel, "g_trainingSubstrate_textureDest", target[destTarget]);
            shader.Dispatch(kernel, Math.Max(1, textureSize / 8), Math.Max(1, textureSize / 8), 1);
        }

        kernel = shader.FindKernel("GenerateTrainingSubstrate_EdgeBlur");
        destTarget = 1 - destTarget;
        shader.SetTexture(kernel, "g_trainingSubstrate_textureRef", target[1-destTarget]);
        shader.SetTexture(kernel, "g_trainingSubstrate_textureDest", target[destTarget]);
        shader.Dispatch(kernel, Math.Max(1, textureSize / 8), Math.Max(1, textureSize / 8), 1);

        // Add Noise
        kernel = shader.FindKernel("GenerateTrainingSubstrate_AddNoise");
        destTarget = 1 - destTarget;
        shader.SetTexture(kernel, "g_trainingSubstrate_textureRef", target[1-destTarget]);
        shader.SetTexture(kernel, "g_trainingSubstrate_textureDest", target[destTarget]);
        shader.Dispatch(kernel, Math.Max(1, textureSize / 8), Math.Max(1, textureSize / 8), 1);

        // Gradient/Coloration
        kernel = shader.FindKernel("GenerateTrainingSubstrate_Gradient");
        destTarget = 1 - destTarget;
        shader.SetTexture(kernel, "g_trainingSubstrate_textureRef", target[1-destTarget]);
        shader.SetTexture(kernel, "g_trainingSubstrate_textureDest", target[destTarget]);
        shader.Dispatch(kernel, Math.Max(1, textureSize / 8), Math.Max(1, textureSize / 8), 1);

        // Hardness
        kernel = shader.FindKernel("GenerateTrainingSubstrate_Hardness");
        destTarget = 1 - destTarget;
        shader.SetTexture(kernel, "g_trainingSubstrate_textureRef", target[1-destTarget]);
        shader.SetTexture(kernel, "g_trainingSubstrate_textureDest", target[destTarget]);
        shader.Dispatch(kernel, Math.Max(1, textureSize / 8), Math.Max(1, textureSize / 8), 1);

        GameObject.DestroyImmediate(target[1-destTarget]);
        return target[destTarget];
    }

}