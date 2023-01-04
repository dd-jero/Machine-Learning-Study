//MummyAgent가 박스에 충돌 시 Epsoide 종료 및 초기화 (ppo algorithm으로 training)

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

// Agent의 역할 : 1. 주변 환경을 관측(Observation) 2. 정책에 의해 행동(Action) 3. 보상(Reward)

public class MummyAgent : Agent
{
    private new Transform transform;    // Transfome은  오브젝트의 위치, 회전, 크기를 결정하는 컴포넌트
    private new Rigidbody rigidbody;    // Rigidbody는 오브젝트가 물리제어로 동작하도록 함.(힘과 토크를 받아 사실적으로 움직임 ==> 중력의 영향)
    public Transform targetTr;
   
    public Material badMt, goodMt;
    private Material originMt;
    private Renderer floorRd;

    public override void Initialize() // 초기화 메소드
    {
       transform = GetComponent<Transform>();
       rigidbody = GetComponent<Rigidbody>();
       targetTr = transform.parent.Find("Target");
       floorRd = transform.parent.Find("Floor").GetComponent<Renderer>();   
       originMt = floorRd.material;
    }

    public override void OnEpisodeBegin() // 에피소드가 시작될 때마다 호출되는 메소드
    {
        rigidbody.velocity = Vector3.zero;
        rigidbody.angularVelocity = Vector3.zero;
  
        transform.localPosition = new Vector3(Random.Range(-4.0f, 4.0f), 0.05f, Random.Range(-4.0f, 4.0f));
        targetTr.localPosition = new Vector3(Random.Range(-4.0f, 4.0f), 0.55f, Random.Range(-4.0f, 4.0f));

        StartCoroutine(RevertMaterial());
    }

    IEnumerator RevertMaterial()
    {
        yield return new WaitForSeconds(0.2f);
        floorRd.material = originMt;
    }

    public override void CollectObservations(Unity.MLAgents.Sensors.VectorSensor sensor) // 주변 환경을 관측하는 callback 메소드
    {
        sensor.AddObservation(targetTr.localPosition);
        sensor.AddObservation(transform.localPosition);
        sensor.AddObservation(rigidbody.velocity.x);
        sensor.AddObservation(rigidbody.velocity.z);
    }
    
    public override void OnActionReceived(ActionBuffers actions) // 정책으로 전달받은 데이터를 기반으로 행동을 실행하는 메소드
    {
        var action = actions.ContinuousActions;
        Vector3 dir = (Vector3.forward * action[0]) + (Vector3.right * action[1]);
        rigidbody.AddForce(dir.normalized * 50.0f);

        SetReward(-0.001f);
    }


    public override void Heuristic(in ActionBuffers actionsOut) // 개발자의 테스트 용도, 모방학습
    {
        var action = actionsOut.ContinuousActions;

        action[0] = Input.GetAxis("Vertical");
        action[1] = Input.GetAxis("Horizontal");

    }

    void OnCollisionEnter(Collision collision) // 보상 처리 로직 (충돌 시)
    {
        if (collision.collider.CompareTag("DEAD_ZONE"))
        {
            floorRd.material = badMt;
            AddReward(-1.0f);
            EndEpisode();
        }
        if (collision.collider.CompareTag("TARGET"))
        {
            floorRd.material = goodMt;
            AddReward(1.0f);
            EndEpisode();
        }
    }
}
