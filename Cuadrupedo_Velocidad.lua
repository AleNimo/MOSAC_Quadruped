-- sim.adjustView
function createAgent ()
    -- Restore the agent
    agent=sim.loadModel(agentData)

    -- Get agent handles
    joint[ 1] = sim.getObject("/JointBFR")
    joint[ 2] = sim.getObject("/JointLFR")
    joint[ 3] = sim.getObject("/JointPFR")
    joint[ 4] = sim.getObject("/JointBFL")
    joint[ 5] = sim.getObject("/JointLFL")
    joint[ 6] = sim.getObject("/JointPFL")
    joint[ 7] = sim.getObject("/JointBBR")
    joint[ 8] = sim.getObject("/JointLBR")
    joint[ 9] = sim.getObject("/JointPBR")
    joint[10] = sim.getObject("/JointBBL")
    joint[11] = sim.getObject("/JointLBL")
    joint[12] = sim.getObject("/JointPBL")
    tip[1],target[1] = sim.getObject("/TipFR"),sim.getObject("/TargetFR")
    tip[2],target[2] = sim.getObject("/TipFL"),sim.getObject("/TargetFL")
    tip[3],target[3] = sim.getObject("/TipBR"),sim.getObject("/TargetBR")
    tip[4],target[4] = sim.getObject("/TipBL"),sim.getObject("/TargetBL")
    
    --sim.shapeintparam_respondable_mask
    --local res,collPair=sim.checkCollision(h,sim.handle_all)
    
    -- Create the inverse kinematics environment
    ikEnv=simIK.createEnvironment()
    ikGroup=simIK.createGroup(ikEnv)
    for i=1,leg_number,1 do
        simIK.addElementFromScene(ikEnv,ikGroup,agent,tip[i],target[i],simIK.constraint_position)
    end
    
    -- Get the agent initial status
    agentObjects=sim.getObjectsInTree(agent,sim.handle_all,0)
    agentInitialConfig=sim.getConfigurationTree(agent)
    
    -- Define the programmed policy's base position
    x0, y0, z_rise = 0.0, 0.0, -0.002
    base_pos = { sim.getObjectPosition(tip[1], agent), sim.getObjectPosition(tip[2], agent), sim.getObjectPosition(tip[3], agent), sim.getObjectPosition(tip[4], agent) }
    base_pos[1] = { base_pos[1][1]+x0, base_pos[1][2]-y0, base_pos[1][3]+(x0+y0)/2 }
    base_pos[2] = { base_pos[2][1]+x0, base_pos[2][2]+y0, base_pos[2][3]+(x0+y0)/2 }
    base_pos[3] = { base_pos[3][1]-x0, base_pos[3][2]-y0, base_pos[3][3]+(x0+y0)/2 }
    base_pos[4] = { base_pos[4][1]-x0, base_pos[4][2]+y0, base_pos[4][3]+(x0+y0)/2 }
    walk_pos = { { base_pos[1], base_pos[2], base_pos[3], base_pos[4] }, { base_pos[1], base_pos[2], base_pos[3], base_pos[4] } }
    for i=1,leg_number,1 do sim.setObjectPosition(target[i], agent, base_pos[i]) end
    simIK.applyIkEnvironmentToScene(ikEnv,ikGroup)

end

function destroyAgent ()
    simIK.eraseEnvironment(ikEnv)
    sim.removeModel(agent)
end

function sysCall_init() -- Executed when the scene is loaded
    local HOST, PORT = "127.0.0.1", 57175
    local socket = require('socket')

    -- Create the client and initial connection
    client, err = socket.connect(HOST, PORT)
    client:setoption('keepalive', true)
    Rx_float_length = 10
    Tx_float_length = "%010.5f"
    Tx_Rx_command_length = 5
    
    -- Reset position
    reset_pos = {0.0, 0.0, 0.233}
    reset_or = {0.0, 0.0, 0.0}
    
    n = 5.0   -- Number of intermediate frames between two main frames in direction control
    mode = 0  -- mode 0 = joint control / 1 = direction control

    -- Joints' absolut limits
--    bod_llim, bod_ulim = -15.0*math.pi/180.0,  60.0*math.pi/180.0
--    leg_llim, leg_ulim = -40.0*math.pi/180.0, 130.0*math.pi/180.0
--    paw_llim, paw_ulim = -90.0*math.pi/180.0,  30.0*math.pi/180.0
    -- Joints' prefered limits
    bod_llim, bod_ulim = -10.0*math.pi/180.0, 15.0*math.pi/180.0
    leg_llim, leg_ulim = -10.0*math.pi/180.0, 40.0*math.pi/180.0
    paw_llim, paw_ulim = -15.0*math.pi/180.0,  5.0*math.pi/180.0
    jointLowerLimit = {bod_llim, leg_llim, paw_llim, bod_llim, leg_llim, paw_llim, bod_llim, leg_llim, paw_llim, bod_llim, leg_llim, paw_llim}
    jointUpperLimit = {bod_ulim, leg_ulim, paw_ulim, bod_ulim, leg_ulim, paw_ulim, bod_ulim, leg_ulim, paw_ulim, bod_ulim, leg_ulim, paw_ulim}
    max_delta = 0.01   -- Newton
--    max_delta = 0.1   -- Bullet
    max_delta_t = 0.02

    -- Get agent handles
    joints_number = 12
    leg_number = 4
    joint = {}
    tip = {}
    target = {}
    
    -- Load the agent's model
    agent=sim.loadModel(sim.getStringParam(sim.stringparam_scene_path)..'/Quadruped_corregido.ttm')

    -- Save the agent's model
    agentData = sim.saveModel(agent)
    sim.removeModel(agent)
    
    base_pos, relative_pos = {}, {}
    state = 0 -- state 0 = idle / 1 = moving to intermediate position / 2 = moving to target position / 3 = reset
    r = 0.025
end

function sysCall_beforeSimulation() -- Executed just before the simulation starts
    -- Load the agent model
    createAgent()

    -- Get the agent initial status
    sim.setObjectPosition(agent,-1,reset_pos)
    sim.setObjectOrientation(agent,-1,reset_or)

    -- Define joints' position variables
    jointPos = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    jointMidPos = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    jointTargetPos = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    lastJointPos = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
--    walk_dir = {1.0, 1.0, 1.0, 1.0}
    f1, f2, f = 1, 2, 0.0      -- Movement frames
    sequence_step = 0
    state = 0
end

function sysCall_actuation()
    if state == 3 then -- If state is 'reset'
        -- Do nothing
    elseif state == 0 then -- If state is 'idle'
        --sim.pauseSimulation()
        --Send the agent's status
        --Obtain and send the object world position (change the second parameter from -1 to another handle to get a relative position)
        data = sim.getObjectPosition(agent, -1)
        client:send(string.format(Tx_float_length, data[1]))
        client:send(string.format(Tx_float_length, data[2]))
        client:send(string.format(Tx_float_length, data[3]))
        --Obtain and send the object world rotation (change the second parameter from -1 to another handle to get a relative position)
        data = sim.getObjectOrientation(agent, -1)
        client:send(string.format(Tx_float_length, data[1]/math.pi))
        client:send(string.format(Tx_float_length, data[2]/math.pi))
        --For the Z angle that has full rotation we send the cosine and sine to avoid the discontinuity
        client:send(string.format(Tx_float_length, math.cos(data[3])))
        client:send(string.format(Tx_float_length, math.sin(data[3])))
        --Send the joints positions
        for i=1,joints_number,1 do
            jointPos[i] = (sim.getJointPosition(joint[i]) - (jointUpperLimit[i]+jointLowerLimit[i])/2) / ((jointUpperLimit[i]-jointLowerLimit[i])/2)
            client:send(string.format(Tx_float_length, jointPos[i]))
        end
        --Obtain and send the object world velocity
        data, _ = sim.getObjectVelocity(agent)
        client:send(string.format(Tx_float_length, data[1]))
        client:send(string.format(Tx_float_length, data[2]))
        client:send(string.format(Tx_float_length, data[3]))

        --Receive the agent's next action
        data = {}
        data, status, partial_data = client:receive(Tx_Rx_command_length)
        --print(data)
        if data=="RESET" then
            --print("RESET: Environment reset")
            --Get new position
            reset_pos[1], status, partial_data = client:receive(Rx_float_length)
            reset_pos[2], status, partial_data = client:receive(Rx_float_length)
            reset_or[3], status, partial_data = client:receive(Rx_float_length)
            reset_or[3] = math.pi * reset_or[3]
            --Destroy the agent and restart the simulation
            destroyAgent()
            state = 3
            sim.stopSimulation()

        elseif data=="ACT__" then
            --print("ACT__: Action")
            if mode == 0 then
                --Receive joints target positions and set the midpoint targets
                for i=1,joints_number,1 do
                    data, status, partial_data = client:receive(Rx_float_length)
                    jointTargetPos[i] = ((jointUpperLimit[i]-jointLowerLimit[i])/2) * tonumber(data) + (jointUpperLimit[i]+jointLowerLimit[i])/2
                    jointMidPos[i] = jointTargetPos[i]/2 + sim.getJointPosition(joint[i])/2
                    sim.setJointTargetPosition(joint[i],jointMidPos[i])
                    lastJointPos[i] = jointPos[i]
                end
                last_t = sim.getSimulationTime()
            else
                --Receive joints target positions and set the midpoint targets
                data, status, partial_data = client:receive(Rx_float_length)
                dir = math.pi*tonumber(data)
                dir_x, dir_y = math.cos(dir), math.sin(dir)
                x0, y0 = dir_x*r, dir_y*r
--                print("Dir: "..(dir/math.pi*180).." - Relative pos: ("..x0..", "..y0..")")
                if sequence_step == 1 then
                    walk_pos[f2][1] = { (walk_pos[f1][1][1]+base_pos[1][1]-x0)/2, (walk_pos[f1][1][2]+base_pos[1][2]-y0)/2, base_pos[1][3]-z_rise }
                    walk_pos[f2][2] = { (walk_pos[f1][2][1]+base_pos[2][1]+x0)/2, (walk_pos[f1][2][2]+base_pos[2][2]+y0)/2, base_pos[1][3]+z_rise }
                    walk_pos[f2][3] = { (walk_pos[f1][3][1]+base_pos[3][1]+x0)/2, (walk_pos[f1][3][2]+base_pos[3][2]+y0)/2, base_pos[1][3]+z_rise }
                    walk_pos[f2][4] = { (walk_pos[f1][4][1]+base_pos[4][1]-x0)/2, (walk_pos[f1][4][2]+base_pos[4][2]-y0)/2, base_pos[1][3]-z_rise }
                    sequence_step = 2
                elseif sequence_step == 2 then
                    walk_pos[f2][1] = { base_pos[1][1]-x0, base_pos[1][2]-y0, base_pos[1][3]-z_rise }
                    walk_pos[f2][2] = { base_pos[2][1]+x0, base_pos[2][2]+y0, base_pos[1][3]-z_rise }
                    walk_pos[f2][3] = { base_pos[3][1]+x0, base_pos[3][2]+y0, base_pos[1][3]-z_rise }
                    walk_pos[f2][4] = { base_pos[4][1]-x0, base_pos[4][2]-y0, base_pos[1][3]-z_rise }
                    sequence_step = 3
                elseif sequence_step == 3 then
                    walk_pos[f2][1] = { (walk_pos[f1][1][1]+base_pos[1][1]+x0)/2, (walk_pos[f1][1][2]+base_pos[1][2]+y0)/2, base_pos[1][3]+z_rise }
                    walk_pos[f2][2] = { (walk_pos[f1][2][1]+base_pos[2][1]-x0)/2, (walk_pos[f1][2][2]+base_pos[2][2]-y0)/2, base_pos[1][3]-z_rise }
                    walk_pos[f2][3] = { (walk_pos[f1][3][1]+base_pos[3][1]-x0)/2, (walk_pos[f1][3][2]+base_pos[3][2]-y0)/2, base_pos[1][3]-z_rise }
                    walk_pos[f2][4] = { (walk_pos[f1][4][1]+base_pos[4][1]+x0)/2, (walk_pos[f1][4][2]+base_pos[4][2]+y0)/2, base_pos[1][3]+z_rise }
                    sequence_step = 4
                else
                    walk_pos[f2][1] = { base_pos[1][1]+x0, base_pos[1][2]+y0, base_pos[1][3]-z_rise }
                    walk_pos[f2][2] = { base_pos[2][1]-x0, base_pos[2][2]-y0, base_pos[1][3]-z_rise }
                    walk_pos[f2][3] = { base_pos[3][1]-x0, base_pos[3][2]-y0, base_pos[1][3]-z_rise }
                    walk_pos[f2][4] = { base_pos[4][1]+x0, base_pos[4][2]+y0, base_pos[1][3]-z_rise }
                    sequence_step = 1
                end
--                for l=1,leg_number,1 do
--                    print("LEG "..l)
--                    print("base: { "..base_pos[l][1]..", "..base_pos[l][2]..", "..base_pos[l][3].." }")
--                    print("pos1: { "..walk_pos[f1][l][1]..", "..walk_pos[f1][l][2]..", "..walk_pos[f1][l][3].." }")
--                    print("pos2: { "..walk_pos[f2][l][1]..", "..walk_pos[f2][l][2]..", "..walk_pos[f2][l][3].." }")
--                end
            end
            --Signal midpoint flag
            state = 1
        else
            --print("MODE_: Mode change")
            --Receive new mode
            data, status, partial_data = client:receive(Rx_float_length)
            mode = tonumber(data)
        end

    elseif mode == 0 then -- If state is not 'idle' and mode is joint control

        -- Get the current position of all the joints
        for i=1,joints_number,1 do
            jointPos[i] = sim.getJointPosition(joint[i])
        end
        
        -- Get the current time and if a certain delta passed with no change, end movement
        t = sim.getSimulationTime()
        if t > last_t + max_delta_t then
            last_t = t
            last_state, state = state, 0
            for i=1,joints_number,1 do
                delta = math.abs(jointPos[i] - lastJointPos[i])
--                print(i, delta, jointPos[i], lastJointPos[i])
                lastJointPos[i] = jointPos[i]
                if delta >= 2*math.pi then delta = delta - 2*math.pi end
                if delta > max_delta then state = last_state end
            end
--            print("Timeout state:", state)
        end
        
        if state == 1 then
            -- Release the midpoint flag if all joints are within a delta error
            state = 2
            for i=1,joints_number,1 do
                delta = math.abs(jointPos[i] - jointMidPos[i])
                if delta >= 2*math.pi then delta = delta - 2*math.pi end
                if delta > max_delta then state = 1 end
            end
            -- If so, set the final target position for all joints
            if state == 2 then
--                print("MidPoint reached")
                for i=1,joints_number,1 do
                    sim.setJointTargetPosition(joint[i], jointTargetPos[i])
                    lastJointPos[i] = jointPos[i]
                end
                last_t = t
            end
        elseif state == 2 then
            -- Check if all joint are within a delta error of the final target
            state = 0
            for i=1,joints_number,1 do
                delta = math.abs(jointPos[i] - jointTargetPos[i])
                if delta >= 2*math.pi then delta = delta - 2*math.pi end
                if delta > max_delta then state = 2 end
            end
--            if state == 0 then
--                print("TargetPos reached")
--            end
        end

    else -- If state is not 'idle' and mode is direction control

        if f > n then
            state = 0
            f1, f2 = f2, f1
            f = 0.0
        else
            -- Execute coded leg movement (lineal interpolation between two frames)
            for l=1,leg_number,1 do
                relative_pos[1] = walk_pos[f1][l][1] * (n-f)/n + walk_pos[f2][l][1] * f/n
--                relative_pos[2] = walk_dir[l] * (walk_pos[f1][l][2] * (n-f)/n + walk_pos[f2][l][2] * f/n)
                relative_pos[2] = walk_pos[f1][l][2] * (n-f)/n + walk_pos[f2][l][2] * f/n
                relative_pos[3] = walk_pos[f1][l][3] * (n-f)/n + walk_pos[f2][l][3] * f/n
                sim.setObjectPosition(target[l],agent,relative_pos)
            end
            simIK.applyIkEnvironmentToScene(ikEnv,ikGroup)
            f = f + 1
        end

    end

end

function sysCall_afterSimulation() -- Executed just before the simulation ends
    -- Destroy the inverse kinematics environment
    if state ~= 3 then simIK.eraseEnvironment(ikEnv) end
end

function sysCall_cleanup() -- Executed when the scene is closed
    -- Close the communication socket
    client:close()
end

function sysCall_nonSimulation() -- Executed when the simulation is not running
    -- Restart the simulation
    if state == 3 then sim.startSimulation() end
end

-- See the user manual or the available code snippets for additional callback functions and details
