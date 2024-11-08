using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

[RequireComponent(typeof(TMP_Text))]
public class SimulationPerfDisplay : MonoBehaviour
{
    public enum PerfDisplayType {
        TraversalsPerSecond,
        ConvergenceValue,
        ConvergenceTime
    }

    [SerializeField] private Simulation simulation;
    [SerializeField] private PerfDisplayType displayData;

    void Start()
    {
        if(!simulation)
            Debug.Log("simulation property is not set on SimulationPerfDisplay");
    }

    // Update is called once per frame
    void Update()
    {
        string value = "";
        bool doUpdate = true;
        switch(displayData) {
        case PerfDisplayType.TraversalsPerSecond:
            doUpdate = !simulation.hasConverged;
            value = (simulation.TraversalsPerSecond / 1000000.0f).ToString("0.0") + " MTPS";
            break;
        case PerfDisplayType.ConvergenceValue:
            value = simulation.Convergence.ToString() + " ξ";
            break;
        case PerfDisplayType.ConvergenceTime:
            doUpdate = !simulation.hasConverged;
            value = (Time.time - simulation.ConvergenceStartTime).ToString("0.0") + "s";
            break;
        }

        if(simulation && doUpdate) {
            GetComponent<TMP_Text>().text = value;
        }
    }
}
