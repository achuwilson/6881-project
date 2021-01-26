#!/usr/bin/python3

from pydrake.manipulation.simple_ui import JointSliders
from pydrake.systems.framework import(DiagramBuilder,
     LeafSystem, BasicVector, PublishEvent, TriggerType)
from pydrake.systems.analysis import Simulator
from pydrake.systems.primitives import FirstOrderLowPassFilter
from iiwa_manipulation_station import IiwaManipulationStation
import numpy as np
import matplotlib.pyplot as plt
from pydrake.systems.drawing import plot_system_graphviz, plot_graphviz
from pydrake.manipulation.planner import (
    DifferentialInverseKinematicsParameters)
from pydrake.all import (JacobianWrtVariable, SystemSliders, Integrator,
     PortDataType,TriggerType, Parser, LogOutput,DiscreteDerivative)
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
#from differential_ik import DifferentialIK
import circle_fit as cf



#import matplotlib as mpl
#mpl.rcParams['figure.dpi'] = 1200    
import time
import lcm
from drake import lcmt_iiwa_status

#initiate the LCM instance
lc = lcm.LCM()

#this is used to subscribe to LCM mesages
class lcm_subscriptor(object):
    def __init__(self, channel, lcm_type, lc):
        self.subscriptor = lc.subscribe(channel,self._msg_handle)
        self.lcm_type = lcm_type
        self.msg = lcm_type()
    def _msg_handle(self, channel, message):
        self.msg = self.lcm_type.decode(message)

#we subscribe to "IIWA_STATUS" LCM message to set the initial values of the 
# drake systems, so that it matches the hardware values        
subscription = lcm_subscriptor("IIWA_STATUS",lcmt_iiwa_status,lc)
lc.handle()
#define a plant which will just print the data coming into the input port
class PrintSystem(LeafSystem):
    def __init__(self, num_input = 1):
        LeafSystem.__init__(self)
        self.set_name('PrintSystem')
        self.input_port = self.DeclareInputPort(
            "data_in", PortDataType.kVectorValued, num_input)

        #Declare a periodic event that updates at 100Hz
        self.DeclarePeriodicEvent(period_sec =1.0/100,
                                 offset_sec=0.00,
                                 event=PublishEvent(
                                     trigger_type=TriggerType.kPeriodic,
                                     callback=self._periodic_update))

    def _periodic_update(self, context, event):
        #the periodic callback function
        #read data from the input port
        msg = self.input_port.Eval(context)
        print(msg)

def circleEstimator(A,B,C):
    a = np.linalg.norm(C - B)
    b = np.linalg.norm(C - A)
    c = np.linalg.norm(B - A)
    s = (a + b + c) / 2
    R = a*b*c / 4 / np.sqrt(s * (s - a) * (s - b) * (s - c))
    b1 = a*a * (b*b + c*c - a*a)
    b2 = b*b * (a*a + c*c - b*b)
    b3 = c*c * (a*a + b*b - c*c)
    P = np.column_stack((A, B, C)).dot(np.hstack((b1, b2, b3)))
    P /= b1 + b2 + b3
    print(" Circle Parameters R, P",R, P)

class BoxPlayer(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self.set_name('BoxPlayer')
        self.robot =  plant
        self.robot_context = plant.CreateDefaultContext()
        #The world frame and the end-effector frame
        self.EE = plant.GetBodyByName("tip").body_frame()
        self.EEb =  plant.GetBodyByName("tip")
        self.WO = plant.world_frame()

        #declare input ports
        self.input_port0 = self.DeclareVectorInputPort("iiwa_position_in",
                                                        BasicVector(7))
        self.input_port1 = self.DeclareVectorInputPort("iiwa_torque_in",
                                                        BasicVector(7))
        self.input_port2 = self.DeclareVectorInputPort("differential_force_in",
                                                        BasicVector(6))                                                

        self.output_port0 =self.DeclareVectorOutputPort("velocity_commanded", 
                                                        BasicVector(7),
                                                        self.CopyStateOutVel) 

        self.output_port1 =self.DeclareVectorOutputPort("torque_commanded", 
                                                        BasicVector(7),
                                                        self.CopyStateOutTorque)                                                                                          
        self.output_port2 =self.DeclareVectorOutputPort("force_calculated", 
                                                        BasicVector(6),
                                                        self.CopyStateOutForce)
        self.output_port3 =self.DeclareVectorOutputPort("position_calculated", 
                                                        BasicVector(3),
                                                        self.CopyStateOutPosition)                                                
        #self.madeContact = False
        self.zForceThres = 6.0
        self.minXpos = 0.47
        self.minZpos = 0.15
        self.zDownForce = np.array([0,0,0,0,0,-8.0])
        self.zDownVel = np.array([0,0,0,0,0,-0.02]) #
        self.xNegVel = np.array([0,0,0,-0.004,0,-0.00])
        self.xPosVel = np.array([0,0,0,0.002,0,-0.00])
        self.xNegVelSlow = np.array([0,0,0,-0.0015,0,-0.00])
        self.zDownVelEdge = np.array([0,0,0,0,0,-0.002])
        self.DeclareDiscreteState([0])
        time_step =0.005
        self.DeclarePeriodicDiscreteUpdate(time_step)
        #self.P1 = None
        #self.P2 = None
        #self.P3 = None
        #self.P1t = None
        #self.P2dt = 2.0
        self.circlepoints = []
        print("BOXPLAYER INIT OVER")
        # STATES
        # -1 : STOP
        #  0 : Moving Down
        #  1 : Moving to edge at xNegVel
        # 1.5 : intialize pivoting
        #  2 : at 0 fZ position, balancing

    def DoCalcDiscreteVariableUpdates(
        self, context, events, discrete_state):
        #state 0 - precontact
        #get the current state
        current_state = context.get_discrete_state_vector().GetAtIndex(0)
        #update joint positions of multibodyplant
        q_pos = self.input_port0.Eval(context)
        self.robot.SetPositions(self.robot_context,q_pos)
        q_torque = self.input_port1.Eval(context)
        J = self.robot.CalcJacobianSpatialVelocity(self.robot_context,
                                                    JacobianWrtVariable.kQDot,
                                                    self.EE,
                                                    [0,0,0],
                                                    self.WO,
                                                    self.WO)
        Wext = np.linalg.pinv(J.T).dot(q_torque)
        eepos = self.robot.EvalBodyPoseInWorld(self.robot_context, self.EEb)
        dFZ = self.input_port2.Eval(context)[5]
        #print("EE Pos Z, Fx, Fz, st,dfz:",eepos.translation(),"---",Wext[3],Wext[5], current_state,dFZ)
        

        
        if(current_state ==0):
            #estiimate joint_forces from input
            #print( "Z Force is ", Wext[5], "---", current_state)
            #if above threshold z force, set state as 1
            if(Wext[5]>=self.zForceThres):
                discrete_state.get_mutable_vector().SetAtIndex(0, 1)
                print("STATE 1")

        elif(current_state ==1):
            #evaluate position of gripper in world
            eepos = self.robot.EvalBodyPoseInWorld(self.robot_context, self.EEb)
            if(eepos.translation()[0]<=self.minXpos):
                current_state= -1
            if(eepos.translation()[2]<=self.minZpos):
                current_state= -1  
            if((dFZ>0.5) and(context.get_time()>15)):
                current_state = 1.5
                print("STATE 1.5")
                self.P1t = context.get_time()
                self.P1 = eepos.translation() 
                print("sampled first point", self.P1)       
            
            #print( "Z Force is ", Wext[5], "---", current_state)
            discrete_state.get_mutable_vector().SetAtIndex(0, current_state)
        elif(current_state==1.5):

            #if(context.get_time()-self.P1t ==self.P2dt):
            #    self.P2 = eepos.translation()
            #    print("sampled second point", self.P2)
            self.circlepoints.append((eepos.translation()[0],eepos.translation()[2]))
            if(Wext[5]<=0):
                current_state= 2
                #self.P3 = eepos.translation()
                #print("sampled third point", self.P3)
                print("CIRCLE PARAMS1",cf.least_squares_circle(self.circlepoints))
                print("CIRCLE PARAMS2",cf.hyper_fit(self.circlepoints))

                print("STATE :2 ")

                #circleEstimator(self.P1,self.P2,self.P3)
            discrete_state.get_mutable_vector().SetAtIndex(0, current_state)    
        elif(current_state==2):
            #eepos = self.robot.EvalBodyPoseInWorld(self.robot_context, self.EEb)
            #print("EE Pos Z, Fx, Fz :",eepos.translation()[2],Wext[3],Wext[5] ) 
            if(Wext[3]<=0): #stop when x force =0   
                current_state==3
            discrete_state.get_mutable_vector().SetAtIndex(0, current_state)   
        else:
            discrete_state.get_mutable_vector().SetAtIndex(0, current_state)

    
    def CopyStateOutVel(self,context,output):
        #output.SetFromVector(context.get_discrete_state_vector().get_value())
        current_state = context.get_discrete_state_vector().GetAtIndex(0)
        q_pos = self.input_port0.Eval(context)
        q_torque = self.input_port1.Eval(context)
        self.robot.SetPositions(self.robot_context,q_pos)
        #evaluate the Jacobian
        J = self.robot.CalcJacobianSpatialVelocity(self.robot_context,
                                                JacobianWrtVariable.kQDot,
                                                self.EE,
                                                [0,0,0],
                                                self.WO,
                                                self.WO)

        if(current_state ==0):
            q_vel = np.linalg.pinv(J).dot(self.zDownVel)
        elif(current_state ==1):
            q_vel = np.linalg.pinv(J).dot(self.xNegVel)
        elif(current_state ==1.5):
            q_vel = np.linalg.pinv(J).dot(self.xNegVel)    
        elif(current_state ==2):
            #continue moving forward
            #q_vel = np.linalg.pinv(J).dot(self.xNegVelSlow)
            #stop at zero z force    
            q_vel = np.zeros(7)
            #go reverse
            #q_vel = np.linalg.pinv(J).dot(self.xPosVel) 
        elif(current_state==3):
            q_vel = np.linalg.pinv(J).dot(self.xPosVel)    
        else:
            q_vel = np.zeros(7)

        output.SetFromVector(q_vel)    
            
    def CopyStateOutTorque(self, context,output):
        current_state = context.get_discrete_state_vector().GetAtIndex(0)
        q_pos = self.input_port0.Eval(context)
        self.robot.SetPositions(self.robot_context,q_pos)
        #evaluate the Jacobian
        J = self.robot.CalcJacobianSpatialVelocity(self.robot_context,
                                                JacobianWrtVariable.kQDot,
                                                self.EE,
                                                [0,0,0],
                                                self.WO,
                                                self.WO)
        if(current_state ==0):
            output.SetFromVector(np.zeros(7))
        elif(current_state==1):
            q_torques = np.dot(J.T, self.zDownForce)
            output.SetFromVector(q_torques)
        elif(current_state==1.5):
            q_torques = np.dot(J.T, self.zDownForce)
            output.SetFromVector(q_torques)    
        elif(current_state==2):
            q_torques = np.dot(J.T, self.zDownForce)
            output.SetFromVector(q_torques)
        elif(current_state==3):
            q_torques = np.dot(J.T, self.zDownForce)
            output.SetFromVector(q_torques)       
        else:
            output.SetFromVector(np.zeros(7)) 

    def CopyStateOutForce(self, context,output):
        q_pos = self.input_port0.Eval(context)
        self.robot.SetPositions(self.robot_context,q_pos)
        q_torque = self.input_port1.Eval(context)
        J = self.robot.CalcJacobianSpatialVelocity(self.robot_context,
                                                    JacobianWrtVariable.kQDot,
                                                    self.EE,
                                                    [0,0,0],
                                                    self.WO,
                                                    self.WO)
        Wext = np.linalg.pinv(J.T).dot(q_torque)
        output.SetFromVector(Wext)
        #print( "Z Force is ", Wext[5], "---", current_state)
    def CopyStateOutPosition(self,context,output):
        q_pos = self.input_port0.Eval(context)
        self.robot.SetPositions(self.robot_context,q_pos)
        eepos = self.robot.EvalBodyPoseInWorld(self.robot_context, self.EEb)
        output.SetFromVector(eepos.translation())

def main():
    builder = DiagramBuilder()

    ######## ADD SYSTEMS #############
    station = builder.AddSystem(IiwaManipulationStation())
    robot = station.get_controller_plant()

    # add the URDF model of the end effector to the MultibodyPlant
    finger = Parser(robot).AddModelFromFile("models/onefinger.urdf", "simplefinger")
    # compute the transform between the base of finger and iiwa_link_7
    X_7G = RigidTransform(RollPitchYaw(0, 0, 0), [0, 0, 0.045])
    #fix the finger to the iiwa_link_7
    robot.WeldFrames(robot.GetFrameByName("iiwa_link_7")
                    ,robot.GetFrameByName("finger_base", finger),
                     X_7G)
    
    station.Finalize()
    station.Connect()

    controller = builder.AddSystem(BoxPlayer(robot)) 
    integrator = builder.AddSystem(Integrator(7))
    force_differential = builder.AddSystem(DiscreteDerivative(num_inputs = 6, 
                                            time_step = 2.0,
                                         	suppress_initial_transient =True))
    pos_differential = builder.AddSystem(DiscreteDerivative(num_inputs = 3, 
                                            time_step = 0.1,
                                         	suppress_initial_transient =True))                                         
    #printer = builder.AddSystem(PrintSystem(1))      

     ######### MAKE CONNECTIONS ############
    builder.Connect(station.GetOutputPort("iiwa_position_measured"),
                        controller.GetInputPort("iiwa_position_in"))
    builder.Connect(station.GetOutputPort("iiwa_torque_external"),
                        controller.GetInputPort("iiwa_torque_in"))
    builder.Connect(controller.GetOutputPort("velocity_commanded"),
                        integrator.get_input_port())
    builder.Connect(controller.GetOutputPort("torque_commanded"),
                        station.GetInputPort("iiwa_feedforward_torque"))                    
    builder.Connect(integrator.get_output_port(),
                        station.GetInputPort("iiwa_position"))   
    builder.Connect(controller.GetOutputPort("force_calculated"),
                        force_differential.get_input_port())
    builder.Connect(controller.GetOutputPort("position_calculated"),
                        pos_differential.get_input_port())  
    builder.Connect(force_differential.get_output_port(),
                        controller.GetInputPort("differential_force_in"))                                      
    ####logger####
    log = LogOutput(controller.GetOutputPort("force_calculated"), builder)                                  
    log2 = LogOutput(force_differential.get_output_port(), builder) 
    log3 = LogOutput(controller.GetOutputPort("position_calculated"), builder)
    log4 = LogOutput(pos_differential.get_output_port(), builder)
    ######### BUILD ############
    diagram = builder.Build()
    simulator = Simulator(diagram)
    
    ######### SET INTIAL CONDITIONS/PARAMETERS ###########
    # This is important to avoid duplicate publishes to the hardware interface:
    simulator.set_publish_every_time_step(False)

    station_context = diagram.GetMutableSubsystemContext(
        station, simulator.get_mutable_context())

    #station.GetInputPort("iiwa_feedforward_torque").FixValue(
    #    station_context, np.zeros(7))

    #get the initial values from he hardware
    lc.handle()
    initPos= list(subscription.msg.joint_position_measured)
    
    #station.GetInputPort("iiwa_position").FixValue(
    #    station_context, initPos)
    #THIS IS IMPORTANTplt.legend()
    # it as the initial pose of the controller 
    integrator.GetMyContextFromRoot(
        simulator.get_mutable_context(
                    )).get_mutable_continuous_state_vector(
                                                ).SetFromVector(initPos)
    

    simulator.set_target_realtime_rate(1.0)

    ############ RUN ##############
    run_duration = 50
    simulator.AdvanceTo(run_duration)
    t = log.sample_times()
    fx = log.data()
    t2 = log2.sample_times()
    fx2= log2.data()
    px = log3.data()
    t3 = log3.sample_times()
    dpx = log4.data()
    t4 = log4.sample_times()
    plt.figure(figsize=(18, 12), dpi=400)
    #plt.plot(t,fx[0], label = 'R')
    #plt.plot(t,fx[1], label = 'P')
    #plt.plot(t,fx[2], label = 'Y')
    #plt.plot(t,fx[3], label = 'fX')
    #plt.plot(t,x[4], label = 'fY')
    #plt.plot(t,fx[5], label = 'fZ')
    #plt.plot(t2,fx2[3], label='dfX')
    plt.plot(t2,fx2[5], label='dfZ')
    #plt.plot(t3,px[0], label = "pX")
    #plt.plot(t3,px[2], label = "pZ")
    #plt.plot(t3,px[0], label = "dpX")
    #plt.plot(t4,dpx[2], label = "dpZ")

    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()

