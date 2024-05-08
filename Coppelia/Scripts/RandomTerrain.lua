function sysCall_init()
    -- do some initialization here    
    
    h=sim.getObject('.')
    
    f = 25/3
    
    px = 0
    py = 0
    
    max_height = 0.04 --(minimum height = 0)

    points = 100    --Too much points makes the agent go through the heightfield shape (bug)

    length = 6 --meters

    resolution = length/points

    -- bit-coded options for HeightfieldShape: 

    -- bit0 set (1): back faces are culled
    -- bit1 set (2): overlay mesh is visible
    -- bit2 set (4): a simple shape is generated instead of a heightfield
    -- bit3 set (8): the heightfield is not respondable

    options = 2

    -- Initialize the pseudo random number generator for the random phases
    math.randomseed( os.time() )
    math.random(); math.random(); math.random()
end

function sysCall_nonSimulation()
    -- is executed when simulation is not running
end

function sysCall_beforeSimulation()
    -- is executed before a simulation starts

    -- px = 2*math.pi * math.random()
    -- py = 2*math.pi * math.random()

    local heights = {}
    for y=1, points, 1 do
        y_ = y*resolution

        for x=1, points, 1 do
            
            x_ = x*resolution
            
            -- a = 0.5 * (math.sin(2*math.pi*f*x_ + 2*math.pi*px) + math.sin(2*math.pi*f*y_ + 2*math.pi*py))
            a = 0.5 * (math.sin(2*math.pi*f*x_+math.pi/2) + math.sin(2*math.pi*f*y_+math.pi/2))

            n_height = max_height/2 * (a + 1)
            
            table.insert(heights, n_height)
        end
    end
    
    terrainShape=sim.createHeightfieldShape(options,0,points,points,length,heights)
    sim.setEngineFloatParam(sim.newton_body_staticfriction, terrainShape, 1)
    sim.setEngineFloatParam(sim.newton_body_kineticfriction, terrainShape, 1)
    sim.setObjectParent(terrainShape,h,true)
    
    handle_objects = {}
    handle_objects[1] = terrainShape

    heights = nil
end

function sysCall_afterSimulation()
    -- is executed before a simulation ends
    sim.removeObjects(handle_objects)
    handle_objects = nil
    -- collectgarbage()
end

function sysCall_cleanup() -- Executed when the scene is closed
    -- do some clean-up here
end

-- See the user manual or the available code snippets for additional callback functions and details
