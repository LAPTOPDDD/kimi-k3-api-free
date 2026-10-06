
"""
Плетёный стул — Geometry Nodes для Blender 4.x
==============================================

Скрипт строит нод-группу геометрических узлов, генерирующую стул:

* сиденье — переплетённые полосы (полотняное переплетение);
* 4 деревянные ножки по углам;
* спинка из вертикальных планок с верхней перекладиной.

Параметры (панель модификатора Geometry Nodes):
    Ширина          — габарит сиденья по X и Y, м;
    Шаг плетения    — расстояние между осями соседних полос, м;
    Толщина полосы  — толщина полос сиденья, м.

Запуск: Text Editor в Blender → Open → Run Script.
После выполнения в сцене появится объект «Плетёный стул».
"""

import bpy
from math import pi

GROUP_NAME = "GN_Плетёный_стул"
OBJECT_NAME = "Плетёный стул"
WOOD_MAT_NAME = "Дерево"
STRIP_MAT_NAME = "Лента"


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def new_socket(ng, name, in_out, socket_type, default=None,
               min_value=None, max_value=None):
    """Создать сокет интерфейса группы (Blender 4.x и 3.6)."""
    if bpy.app.version >= (4, 0, 0):
        sock = ng.interface.new_socket(name=name, in_out=in_out,
                                       socket_type=socket_type)
    else:
        sock = (ng.inputs if in_out == 'INPUT' else ng.outputs).new(
            socket_type, name)
    if default is not None:
        sock.default_value = default
    if min_value is not None:
        sock.min_value = min_value
    if max_value is not None:
        sock.max_value = max_value
    return sock


def add_node(ng, idname, x, y, label=None):
    n = ng.nodes.new(idname)
    n.location = (x, y)
    if label:
        n.label = label
    return n


def link(ng, out_sock, in_sock):
    ng.links.new(out_sock, in_sock)


def math_node(ng, operation, x, y, a=None, b=None, label=None):
    """Узел Math; a/b — число или выходной сокет другого узла."""
    n = add_node(ng, 'ShaderNodeMath', x, y, label)
    n.operation = operation
    for index, value in ((0, a), (1, b)):
        if value is None:
            continue
        if isinstance(value, (int, float)):
            n.inputs[index].default_value = float(value)
        else:
            link(ng, value, n.inputs[index])
    return n.outputs[0]


def combine_xyz(ng, x, y, sx=None, sy=None, sz=None, label=None):
    """Узел Combine XYZ; sx/sy/sz — число или выходной сокет."""
    n = add_node(ng, 'ShaderNodeCombineXYZ', x, y, label)
    for name, value in (('X', sx), ('Y', sy), ('Z', sz)):
        if value is None:
            continue
        if isinstance(value, (int, float)):
            n.inputs[name].default_value = float(value)
        else:
            link(ng, value, n.inputs[name])
    return n.outputs[0]


def make_material(name, color, roughness=0.55):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


# ---------------------------------------------------------------------------
# Построение нод-группы
# ---------------------------------------------------------------------------

def build_node_group(wood_mat, strip_mat):
    old = bpy.data.node_groups.get(GROUP_NAME)
    if old:
        bpy.data.node_groups.remove(old)

    ng = bpy.data.node_groups.new(GROUP_NAME, 'GeometryNodeTree')

    # --- интерфейс группы -------------------------------------------------
    new_socket(ng, "Ширина", 'INPUT', 'NodeSocketFloat',
               default=0.45, min_value=0.2, max_value=1.5)
    new_socket(ng, "Шаг плетения", 'INPUT', 'NodeSocketFloat',
               default=0.05, min_value=0.01, max_value=0.3)
    new_socket(ng, "Толщина полосы", 'INPUT', 'NodeSocketFloat',
               default=0.004, min_value=0.0005, max_value=0.05)
    new_socket(ng, "Геометрия", 'OUTPUT', 'NodeSocketGeometry')

    gi = add_node(ng, 'NodeGroupInput', -2700, 300, "Параметры")
    go = add_node(ng, 'NodeGroupOutput', 2500, 300)

    W = gi.outputs["Ширина"]
    STEP = gi.outputs["Шаг плетения"]
    T = gi.outputs["Толщина полосы"]

    # --- производные величины ---------------------------------------------
    half_w = math_node(ng, 'MULTIPLY', -2500, 900, W, 0.5, "W/2")
    neg_hw = math_node(ng, 'MULTIPLY', -2500, 750, W, -0.5, "-W/2")
    w_div = math_node(ng, 'DIVIDE', -2500, 600, W, STEP, "W/шаг")
    cnt_fl = math_node(ng, 'FLOOR', -2300, 600, w_div, None, "floor")
    count = math_node(ng, 'ADD', -2100, 650, cnt_fl, 1.0, "Число полос")
    span = math_node(ng, 'MULTIPLY', -2100, 500, cnt_fl, STEP, "Пролёт")
    half_sp = math_node(ng, 'MULTIPLY', -1900, 550, span, 0.5, "Пролёт/2")
    start_c = math_node(ng, 'MULTIPLY', -1900, 400, span, -0.5, "Начало")
    strip_w = math_node(ng, 'MULTIPLY', -2500, 400, STEP, 0.8, "Ширина полосы")
    leg_t = math_node(ng, 'MULTIPLY', -2500, 250, W, 0.08, "Сечение ножки")
    back_h = math_node(ng, 'MULTIPLY', -2500, 100, W, 0.9, "Высота спинки")
    slat_w = math_node(ng, 'MULTIPLY', -2500, -50, STEP, 0.5, "Ширина планки")
    slat_t = math_node(ng, 'MULTIPLY', -2500, -200, T, 2.0, "Толщина планки")
    ampl = math_node(ng, 'MULTIPLY', -2500, -350, T, 0.5, "Амплитуда")
    neg_ht = math_node(ng, 'MULTIPLY', -2500, -500, T, -0.5, "-T/2")

    # --- один слой плетения ------------------------------------------------
    def weave_layer(base_y, along_x, phase_shift, label):
        """Слой полос: along_x=True — полосы вдоль оси X, иначе вдоль Y."""
        # точки под копии полос
        ml = add_node(ng, 'GeometryNodeMeshLine', -1600, base_y,
                      label + ": точки")
        ml.mode = 'OFFSET'
        link(ng, count, ml.inputs["Count"])
        if along_x:
            link(ng, combine_xyz(ng, -1850, base_y - 160, 0.0, start_c, W),
                 ml.inputs["Start Location"])
            link(ng, combine_xyz(ng, -1850, base_y - 320, 0.0, STEP, 0.0),
                 ml.inputs["Offset"])
        else:
            link(ng, combine_xyz(ng, -1850, base_y - 160, start_c, 0.0, W),
                 ml.inputs["Start Location"])
            link(ng, combine_xyz(ng, -1850, base_y - 320, STEP, 0.0, 0.0),
                 ml.inputs["Offset"])

        # фаза переплетения: i·π (у второго слоя ещё +π)
        idx = add_node(ng, 'GeometryNodeInputIndex', -1850, base_y + 180)
        ph = math_node(ng, 'MULTIPLY', -1650, base_y + 260,
                       idx.outputs[0], pi, "i·π")
        if phase_shift:
            ph = math_node(ng, 'ADD', -1450, base_y + 300, ph, pi, "i·π+π")

        sna = add_node(ng, 'GeometryNodeStoreNamedAttribute',
                       -1350, base_y, label + ": фаза")
        sna.data_type = 'FLOAT'
        sna.domain = 'POINT'
        sna.inputs["Name"].default_value = "phase"
        link(ng, ml.outputs["Mesh"], sna.inputs["Geometry"])
        link(ng, ph, sna.inputs["Value"])

        # одна полоса: сетка + экструзия на толщину
        grid = add_node(ng, 'GeometryNodeMeshGrid', -1350, base_y - 480,
                        label + ": полоса")
        if along_x:
            link(ng, W, grid.inputs["Size X"])
            link(ng, strip_w, grid.inputs["Size Y"])
            grid.inputs["Vertices X"].default_value = 64
            grid.inputs["Vertices Y"].default_value = 2
        else:
            link(ng, strip_w, grid.inputs["Size X"])
            link(ng, W, grid.inputs["Size Y"])
            grid.inputs["Vertices X"].default_value = 2
            grid.inputs["Vertices Y"].default_value = 64

        ext = add_node(ng, 'GeometryNodeExtrudeMesh', -1150, base_y - 480,
                       label + ": толщина")
        ext.mode = 'FACES'
        link(ng, grid.outputs["Mesh"], ext.inputs["Mesh"])
        link(ng, combine_xyz(ng, -1350, base_y - 700, 0.0, 0.0, T),
             ext.inputs["Offset"])

        # копии полосы по точкам
        iop = add_node(ng, 'GeometryNodeInstanceOnPoints', -950, base_y,
                       label + ": копии")
        link(ng, sna.outputs["Geometry"], iop.inputs["Points"])
        link(ng, ext.outputs["Mesh"], iop.inputs["Instance"])

        rlz = add_node(ng, 'GeometryNodeRealizeInstances', -750, base_y)
        link(ng, iop.outputs["Instances"], rlz.inputs["Geometry"])

        # волна переплетения: z = A·cos(π·(коорд + пролёт/2)/шаг + фаза) - T/2
        pos = add_node(ng, 'GeometryNodeInputPosition', -950, base_y - 520)
        sep = add_node(ng, 'ShaderNodeSeparateXYZ', -750, base_y - 520)
        link(ng, pos.outputs["Position"], sep.inputs[0])
        coord = sep.outputs["X"] if along_x else sep.outputs["Y"]

        na = add_node(ng, 'GeometryNodeInputNamedAttribute',
                      -750, base_y - 780, label + ": чтение фазы")
        na.data_type = 'FLOAT'
        na.inputs["Name"].default_value = "phase"

        s1 = math_node(ng, 'ADD', -550, base_y - 470, coord, half_sp)
        s2 = math_node(ng, 'DIVIDE', -350, base_y - 470, s1, STEP)
        s3 = math_node(ng, 'MULTIPLY', -150, base_y - 470, s2, pi)
        s4 = math_node(ng, 'ADD', 50, base_y - 420, s3,
                       na.outputs["Attribute"])
        s5 = math_node(ng, 'COSINE', 250, base_y - 420, s4, None)
        s6 = math_node(ng, 'MULTIPLY', 450, base_y - 420, s5, ampl)
        s7 = math_node(ng, 'ADD', 650, base_y - 370, s6, neg_ht,
                       "центрирование")

        off = combine_xyz(ng, 850, base_y - 320, 0.0, 0.0, s7)

        sp = add_node(ng, 'GeometryNodeSetPosition', 1050, base_y,
                      label + ": плетение")
        link(ng, rlz.outputs["Geometry"], sp.inputs["Geometry"])
        link(ng, off, sp.inputs["Offset"])

        sm = add_node(ng, 'GeometryNodeSetMaterial', 1250, base_y,
                      label + ": материал")
        sm.inputs["Material"].default_value = strip_mat
        link(ng, sp.outputs["Geometry"], sm.inputs["Geometry"])
        return sm.outputs["Geometry"]

    seat_h = weave_layer(1500, True, False, "Горизонтальные полосы")
    seat_v = weave_layer(400, False, True, "Вертикальные полосы")

    # --- ножки --------------------------------------------------------------
    leg_span = math_node(ng, 'SUBTRACT', -1600, -750, W, leg_t,
                         "Разбег ножек")

    leg_grid = add_node(ng, 'GeometryNodeMeshGrid', -1350, -650,
                        "Точки ножек")
    link(ng, leg_span, leg_grid.inputs["Size X"])
    link(ng, leg_span, leg_grid.inputs["Size Y"])
    leg_grid.inputs["Vertices X"].default_value = 2
    leg_grid.inputs["Vertices Y"].default_value = 2

    leg_size = combine_xyz(ng, -1350, -950, leg_t, leg_t, W)
    leg_cube = add_node(ng, 'GeometryNodeMeshCube', -1150, -850, "Ножка")
    link(ng, leg_size, leg_cube.inputs["Size"])

    leg_shift = combine_xyz(ng, -950, -1050, 0.0, 0.0, half_w)
    leg_tf = add_node(ng, 'GeometryNodeTransform', -950, -850)
    link(ng, leg_cube.outputs["Mesh"], leg_tf.inputs["Geometry"])
    link(ng, leg_shift, leg_tf.inputs["Translation"])

    leg_iop = add_node(ng, 'GeometryNodeInstanceOnPoints', -650, -650,
                       "4 ножки")
    link(ng, leg_grid.outputs["Mesh"], leg_iop.inputs["Points"])
    link(ng, leg_tf.outputs["Geometry"], leg_iop.inputs["Instance"])

    leg_sm = add_node(ng, 'GeometryNodeSetMaterial', -450, -650)
    leg_sm.inputs["Material"].default_value = wood_mat
    link(ng, leg_iop.outputs["Instances"], leg_sm.inputs["Geometry"])

    # --- спинка: вертикальные планки ----------------------------------------
    slat_th = math_node(ng, 'MULTIPLY', -1600, -1250, slat_t, 0.5)
    y_back = math_node(ng, 'ADD', -1400, -1200, neg_hw, slat_th,
                       "Задняя грань")
    back_hh = math_node(ng, 'MULTIPLY', -1600, -1400, back_h, 0.5)
    z_slat = math_node(ng, 'ADD', -1400, -1350, W, back_hh, "Ось планок")

    slat_start = combine_xyz(ng, -1200, -1300, start_c, y_back, z_slat)
    slat_step = combine_xyz(ng, -1200, -1450, STEP, 0.0, 0.0)

    slat_ml = add_node(ng, 'GeometryNodeMeshLine', -1000, -1250,
                       "Точки планок")
    slat_ml.mode = 'OFFSET'
    link(ng, count, slat_ml.inputs["Count"])
    link(ng, slat_start, slat_ml.inputs["Start Location"])
    link(ng, slat_step, slat_ml.inputs["Offset"])

    slat_size = combine_xyz(ng, -1000, -1550, slat_w, slat_t, back_h)
    slat_cube = add_node(ng, 'GeometryNodeMeshCube', -800, -1450, "Планка")
    link(ng, slat_size, slat_cube.inputs["Size"])

    slat_iop = add_node(ng, 'GeometryNodeInstanceOnPoints', -600, -1250,
                        "Планки спинки")
    link(ng, slat_ml.outputs["Mesh"], slat_iop.inputs["Points"])
    link(ng, slat_cube.outputs["Mesh"], slat_iop.inputs["Instance"])

    slat_sm = add_node(ng, 'GeometryNodeSetMaterial', -400, -1250)
    slat_sm.inputs["Material"].default_value = wood_mat
    link(ng, slat_iop.outputs["Instances"], slat_sm.inputs["Geometry"])

    # --- спинка: верхняя перекладина ----------------------------------------
    z_top = math_node(ng, 'ADD', -1400, -1650, W, back_h, "Верх спинки")
    rail_size = combine_xyz(ng, -1000, -1750, W, slat_t, slat_w)
    rail_cube = add_node(ng, 'GeometryNodeMeshCube', -800, -1700,
                         "Перекладина")
    link(ng, rail_size, rail_cube.inputs["Size"])

    rail_shift = combine_xyz(ng, -650, -1850, 0.0, y_back, z_top)
    rail_tf = add_node(ng, 'GeometryNodeTransform', -600, -1650)
    link(ng, rail_cube.outputs["Mesh"], rail_tf.inputs["Geometry"])
    link(ng, rail_shift, rail_tf.inputs["Translation"])

    rail_sm = add_node(ng, 'GeometryNodeSetMaterial', -400, -1650)
    rail_sm.inputs["Material"].default_value = wood_mat
    link(ng, rail_tf.outputs["Geometry"], rail_sm.inputs["Geometry"])

    # --- сборка --------------------------------------------------------------
    join = add_node(ng, 'GeometryNodeJoinGeometry', 2100, 300,
                    "Сборка стула")
    for geom in (seat_h, seat_v,
                 leg_sm.outputs["Geometry"],
                 slat_sm.outputs["Geometry"],
                 rail_sm.outputs["Geometry"]):
        link(ng, geom, join.inputs["Geometry"])
    link(ng, join.outputs["Geometry"], go.inputs["Геометрия"])

    return ng


# ---------------------------------------------------------------------------
# Создание объекта в сцене
# ---------------------------------------------------------------------------

def main():
    # удалить прежний объект, если скрипт уже запускали
    old_obj = bpy.data.objects.get(OBJECT_NAME)
    if old_obj:
        bpy.data.objects.remove(old_obj, do_unlink=True)

    wood = make_material(WOOD_MAT_NAME, (0.32, 0.16, 0.06, 1.0),
                         roughness=0.5)
    strip = make_material(STRIP_MAT_NAME, (0.66, 0.47, 0.24, 1.0),
                          roughness=0.7)

    ng = build_node_group(wood, strip)

    mesh = bpy.data.meshes.new(OBJECT_NAME)
    obj = bpy.data.objects.new(OBJECT_NAME, mesh)
    bpy.context.collection.objects.link(obj)

    mod = obj.modifiers.new(name="Плетёный стул", type='NODES')
    mod.node_group = ng

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    print("Объект «%s» создан. Параметры — в стеке модификаторов."
          % OBJECT_NAME)


if __name__ == "__main__":
    main()
