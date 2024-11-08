using UnityEngine;
using UnityEngine.UIElements;
using UnityEditor;

[CustomEditor(typeof(TrainingSubstrate))]
public class TrainingSubstrateDrawer : Editor
{
    // public override VisualElement CreateInspectorGUI()
    // {
    //     //var inspector = base.CreateInspectorGUI();
    //     // Create a new VisualElement to be the root of our Inspector UI.
    //     VisualElement inspector = new VisualElement();

    //     // Add a simple label.
    //     inspector.Add(new Label("This is a custom Inspector"));

    //     // Return the finished Inspector UI.
    //     return inspector;
    // }

    string seedText;
    string versionText = "2";
    string jsonToLoad;
    public override void OnInspectorGUI()
    {
        TrainingSubstrate substrate = (TrainingSubstrate)target;

        if(substrate.generated) {
            EditorGUI.DrawTextureAlpha(new Rect(10, 10, 200, 200), substrate.generated);
            EditorGUI.DrawPreviewTexture(new Rect(220, 10, 200, 200), substrate.generated);
        }

        GUILayout.Space(220);

        // Draw the default Inspector fields (if needed)
        DrawDefaultInspector();

        if(GUILayout.Button("Generate Random")) {
            substrate.GenerateRandom(version: 2);
        }

        GUILayout.BeginHorizontal();

        if(GUILayout.Button("Generate Seed")) {
            var seed = uint.Parse(seedText, System.Globalization.NumberStyles.HexNumber);
            var version = int.Parse(versionText);
            substrate.GenerateRandom(seed, version);
        }

        seedText = GUILayout.TextField(seedText, GUILayout.Width(100));
        GUILayout.Label(" Ver:", GUILayout.Width(50));
        versionText = GUILayout.TextField(versionText, GUILayout.Width(20));

        GUILayout.EndHorizontal();

        if(GUILayout.Button("Capture JSON (see Log)")) {
            Debug.Log(JsonUtility.ToJson(substrate, true));
        }

        if(GUILayout.Button("Load JSON")) {
            JsonUtility.FromJsonOverwrite(jsonToLoad, substrate);
        }

        jsonToLoad = GUILayout.TextArea(jsonToLoad, GUILayout.MinHeight(20), GUILayout.MaxHeight(100));

        substrate.ValidateAndApply();
    }
}