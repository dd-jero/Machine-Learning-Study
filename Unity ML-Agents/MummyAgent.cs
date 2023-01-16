//MummyAgent가 박스에 충돌 시 Episode 종료 및 초기화 (PPO algorithm으로 training)

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
   
    public Material badMt, goodMt;  // Material은 표면을 렌더링하는 방법을 정의
    private Material originMt;
    private Renderer floorRd;

    public override void Initialize() // 초기화 메소드
    {
       transform = GetComponent<Transform>();       // 이 script가 포함된 오브젝트가 가지고 있는 Transform 컴포넌트를 transform에 저장: GetComponent() 이용
       rigidbody = GetComponent<Rigidbody>();
       targetTr = transform.parent.Find("Target");  // 부모 오브젝트인 "Target"을 찾아서 targetTr에 저장
       floorRd = transform.parent.Find("Floor").GetComponent<Renderer>();   
       originMt = floorRd.material;
    }

    public override void OnEpisodeBegin() // 에피소드(학습단위)가 시작될 때마다 호출되는 메소드
    {
        rigidbody.velocity = Vector3.zero;          // 물리력을 초기화
        rigidbody.angularVelocity = Vector3.zero;
  
        transform.localPosition = new Vector3(Random.Range(-4.0f, 4.0f), 0.05f, Random.Range(-4.0f, 4.0f));     // 에이전트의 위치를 불규칙하게 변경
        targetTr.localPosition = new Vector3(Random.Range(-4.0f, 4.0f), 0.55f, Random.Range(-4.0f, 4.0f));      // 타겟의 위치를 불규칙하게 변경

        StartCoroutine(RevertMaterial()); // IEnumerator를 이용하기 위해 StartCoroutine()을 이용
    }

    IEnumerator RevertMaterial() // IEnumerator (열거자): 작업을 분할하여 수행하는 함수
    {
        // yield 반환문을 이용하는 것 = 일시적으로 CPU 권한을 다른 함수에 위임
        // 권한을 잠시 위임하는 것이기 때문에 다른 함수로 권한을 넘기더라도 자신이 실행하고 있던 상태를 기억하고 있음 => 호출할 때마다 이전에 권한을 위임한 시점부터 다시 코드 실행
        yield return new WaitForSeconds(0.2f); // yield return new WaitForSeconds: 지정된 시간 만큼 대기
        floorRd.material = originMt;
    }

    public override void CollectObservations(Unity.MLAgents.Sensors.VectorSensor sensor) // 환경 정보를 관측/수집해 정책 결정을 위해 브레인에 전달하는 메소드
    {
        sensor.AddObservation(targetTr.localPosition);  // (x,y,z) 3개의 데이터: 타켓의 위치
        sensor.AddObservation(transform.localPosition); // (x,y,z) 3개의 데이터: 에이전트의 위치
        sensor.AddObservation(rigidbody.velocity.x);    // (x) 1개의 데이터: 에이전트의 x축 속도
        sensor.AddObservation(rigidbody.velocity.z);    // (z) 1개의 데이터: 에이전트의 z축 속도 
    }
    
    public override void OnActionReceived(ActionBuffers actions) // 정책으로 전달받은 데이터를 기반으로 행동(Action)을 실행하는 메소드
    {
        // var 타입은 변수의 자료형을 자동으로 저장하지만 지역 변수로 선언해야하며 선언과 동시에 초기화 작업이 필요
        var action = actions.ContinuousActions; // 연속적인 값 (불연속적인 값: actions.DiscreteActions)
        Vector3 dir = (Vector3.forward * action[0]) + (Vector3.right * action[1]); // Vectro3.forward = Vector3(0,0,1), Vector3.right = Vector3(1,0,1)
        rigidbody.AddForce(dir.normalized * 50.0f); // AddForce (연속 + 질량 무시 x): 현실적인 물리현상을 나타낼 때 많이 사용, AddForce(방향.정규화*힘값)
        // 벡터를 정규화 하는 이유는? => 모든 방향의 벡터 길이가 1이어야 방향에 따른 이동속도가 같아진다. (이렇게 정규화된 벡터를 방향벡터)
        
        // SetReward: 이전 보상값을 지우고 현재의 보상값으로 대체 (누적된 보상값이 필요없을 경우 사용)
        SetReward(-0.001f); // 지속적으로 이동을 이끌기 위한 (-) 보상
    }


    public override void Heuristic(in ActionBuffers actionsOut) // 개발자가 직접 명령을 내릴 때 호출하는 메소드 (주로 테스트용, 모방학습에 사용)
    {
        var action = actionsOut.ContinuousActions;

        action[0] = Input.GetAxis("Vertical");      // action[0]에 수평(좌우) 값 입력
        action[1] = Input.GetAxis("Horizontal");    // action[1]에 수직(상하) 값 입력

    }

    void OnCollisionEnter(Collision collision) // 보상 처리 로직 (충돌 시)
    {
        if (collision.collider.CompareTag("DEAD_ZONE")) // 충돌체의 태그가 해당 태그와 같으면 
        {
            floorRd.material = badMt; // floor의 색 변화
            AddReward(-1.0f);   // AddReward: 보상을 받고 바로 에피소드가 종료되지 않고 계속해서 학습해야 하는 환경에서 사용 (잘못된 행동일 때 (-), 올바른 행동일 때 (+) 보상)
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
