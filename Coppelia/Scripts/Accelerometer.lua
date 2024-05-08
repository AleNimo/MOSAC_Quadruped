--lua

sim=require'sim'
simUI=require'simUI'

function handleUI(p)
    local s=sim.getObjectSel()
    if s and #s>0 and s[#s]==model then
        if not ui then
            local xml =[[<ui title="Accelerometer" closeable="false" placement="relative" position="50,-50" layout="form">
                    <label text="x accel:" />
                    <label id="1" text="-" />
                    <label text="y accel:" />
                    <label id="2" text="-" />
                    <label text="z accel:" />
                    <label id="3" text="-" />
            </ui>]]
            ui=simUI.create(xml)
        end
        if p then
            simUI.setLabelText(ui,1,string.format("%.4f",p[1]))
            simUI.setLabelText(ui,2,string.format("%.4f",p[2]))
            simUI.setLabelText(ui,3,string.format("%.4f",p[3]))
        else
            simUI.setLabelText(ui,1,"-")
            simUI.setLabelText(ui,2,"-")
            simUI.setLabelText(ui,3,"-")
        end
    else
        if ui then
            simUI.destroy(ui)
            ui=nil
        end
    end
end

function sysCall_init() 
    model=sim.getObject('.')
    massObject=sim.getObject('./mass')
    sensor=sim.getObject('./forceSensor')
    mass=sim.getObjectFloatParam(massObject,sim.shapefloatparam_mass)
end

function sysCall_sensing() 
    result,force=sim.readForceSensor(sensor)
    if (result>0) then
        accel={force[1]/mass,force[2]/mass,force[3]/mass}
        handleUI(accel)
    else
        handleUI(nil)
    end
end 

function getAccelData()
    return accel
end