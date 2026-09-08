"""NAVSIM 信号灯 / 路口查询：自车实际驶入的那个路口是否灯控、灯色如何。

用**自车真实未来路径**去点查 ROADBLOCK_CONNECTOR（路口内的连接段），
而不是假设 `roadblock_ids` 的顺序 —— 那是 route 规划列表，不保证按距离排序。

`frame["traffic_lights"] = [(lane_connector_id, is_red)]`，`True = RED`
（依据 navsim/planning/scenario_builder/navsim_scenario.py:301）。
"""
from __future__ import annotations
from functools import lru_cache

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

MAP_ROOT = "/data/dataset/navsim/dataset/maps"
MAP_VERSION = "nuplan-maps-v1.0"


@lru_cache(maxsize=8)
def map_api(map_location: str):
    return get_maps_api(MAP_ROOT, MAP_VERSION, map_location)


@lru_cache(maxsize=200000)
def _edges(map_location: str, rbc_id: str):
    """路口连接段的内部 lane connector id 集合。"""
    ob = map_api(map_location).get_map_object(rbc_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)
    return frozenset(e.id for e in ob.interior_edges) if ob is not None else frozenset()


def junction_on_path(map_location: str, path_world_xy):
    """沿自车真实未来路径找它**首个驶入**的路口，返回该点上所有连接段 id 的列表；没有则 None。"""
    m = map_api(map_location)
    for x, y in path_world_xy:
        # 路口内多条连接段重叠很常见，get_one_map_object 会抛异常，必须用 all
        obs = m.get_all_map_objects(Point2D(float(x), float(y)),
                                    SemanticMapLayer.ROADBLOCK_CONNECTOR)
        if obs:
            return [o.id for o in obs]
    return None


def ego_lane_connectors(map_location: str, path_world_xy):
    """自车未来路径**实际经过**的 lane connector id（按沿路径首次出现排序）。

    路口级读数（"任一 connector 绿就算绿"）太弱，会让红灯闸门形同虚设；
    这里把自车匹配到它自己走的那条连接段。
    """
    m = map_api(map_location)
    out = []
    for x, y in path_world_xy:
        for o in m.get_all_map_objects(Point2D(float(x), float(y)),
                                       SemanticMapLayer.LANE_CONNECTOR):
            if o.id not in out:
                out.append(o.id)
    return out


def ego_signal(map_location: str, path_world_xy, traffic_lights):
    """自车进路的灯控状态：(是否灯控, 灯色, 命中的 connector 数)。

    只看**自车自己走的** lane connector 与该帧 traffic_lights 的交集。
    有任一为红 -> RED（保守：进路上有红灯就算受红灯支配）。
    """
    tl = {str(i): bool(r) for i, r in traffic_lights}
    hit = [c for c in ego_lane_connectors(map_location, path_world_xy) if c in tl]
    if not hit:
        return False, None, 0
    return True, ("RED" if any(tl[c] for c in hit) else "GREEN"), len(hit)


def signal_state(map_location: str, rbc_id: str, traffic_lights):
    """返回 (是否灯控, 灯色)。灯色 ∈ {"RED","GREEN",None}。

    该路口的 lane connector 里凡在 traffic_lights 中出现即为**灯控路口**。
    灯色取自车会用到的那些 connector：全红 -> RED；有绿 -> GREEN。
    （自车具体走哪条 connector 需要更细的车道匹配，这里取路口级别的保守读数，
      并在报告中标注这一近似。）
    """
    if not rbc_id:
        return False, None
    ids = [rbc_id] if isinstance(rbc_id, str) else list(rbc_id)
    tl = {str(i): bool(r) for i, r in traffic_lights}
    edges = frozenset().union(*[_edges(map_location, str(x)) for x in ids])
    inter = edges & set(tl)
    if not inter:
        return False, None
    return True, ("RED" if all(tl[x] for x in inter) else "GREEN")
