function sysCall_init()
    -- do some initialization here    
    
    h=sim.getObject('.')
    
    handle_objects = {}
    
    a1 = 1
    a2 = 1
    a3 = 1
    a4 = 1
    
    f1 = 0.5
    f2 = 1
    f3 = 1.5
    f4 = 2
    
    p1 = 0
    p2 = 0
    p3 = 0
    p4 = 0
    
    max_height = 0.15 --(minimum height = 0)

    points = 64

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

    p1 = 2*math.pi * math.random()
    p2 = 2*math.pi * math.random()
    p3 = 2*math.pi * math.random()
    p4 = 2*math.pi * math.random()

    local heights = {}
    for y=1, 64, 1 do
        y_ = y*resolution

        for x=1, 64, 1 do
            
            x_ = x*resolution
            
            a = a1 * 0.5 * (math.sin(2*math.pi*f1*x_ + 2*math.pi*p1) + math.sin(2*math.pi*f1*y_ + 2*math.pi*p1))
            b = a2 * 0.5 * (math.sin(2*math.pi*f2*x_ + 2*math.pi*p2) + math.sin(2*math.pi*f2*y_ + 2*math.pi*p2))
            c = a3 * 0.5 * (math.sin(2*math.pi*f3*x_ + 2*math.pi*p3) + math.sin(2*math.pi*f3*y_ + 2*math.pi*p3))
            d = a4 * 0.5 * (math.sin(2*math.pi*f4*x_ + 2*math.pi*p4) + math.sin(2*math.pi*f4*y_ + 2*math.pi*p4))
            
            n_height = max_height/2 * ((a + b + c + d)/4 + 1)
            
            table.insert(heights, n_height)
        end
    end
    
    terrainShape=sim.createHeightfieldShape(options,0,points,points,length,heights)
    sim.setEngineFloatParam(sim.newton_body_staticfriction, terrainShape, 1)
    sim.setEngineFloatParam(sim.newton_body_kineticfriction, terrainShape, 1)
    sim.setObjectParent(terrainShape,h,true)
    
    table.insert(handle_objects, terrainShape)
end

function sysCall_afterSimulation()
    -- is executed before a simulation ends

    sim.removeObjects(handle_objects)
end

function sysCall_cleanup()
    -- do some clean-up here
end

-- See the user manual or the available code snippets for additional callback functions and details
