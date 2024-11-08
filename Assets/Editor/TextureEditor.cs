using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(RenderTexture))]
public class TextureEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        if(GUILayout.Button("Export")) {
            var path = EditorUtility.SaveFilePanel("Export Texture", ".", "Texture", "png");

            if(!string.IsNullOrEmpty(path)) {
                ((RenderTexture)target).SaveTexturePNG(path);
            }
        }
    }
}