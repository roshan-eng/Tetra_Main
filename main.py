"""
Imported a Custom created class Polyhedron
Import Numpy for handling arrays
"""

import os
import numpy as np
import alphashape
import trimesh
from itertools import combinations
from pathlib import Path
import shutil
import math

from chimerax.core.models import Model
from chimerax.core.commands import run
from chimerax.graphics import Drawing
from chimerax.model_panel import tool

# colors to use
magenta = (255, 0, 255, 255)
cyan = (0, 255, 255, 255)
gold = (255, 215, 0, 255)
bright_green = (70, 255, 0, 255)
navy_blue = (0, 30, 128, 255)
red = (255, 0, 0, 255)
green = (0, 255, 0, 255)
blue = (0, 0, 255, 255)


class Tetra:

    def __init__(self, session):
        self.all_points = None
        self.session = session
        self.model_list = None
        self.t = Drawing('tetrahedrons')
        self.va = []
        self.ta = []
        self.massing_vertices = []
        self.avg_edge_length = 0

        self.model_list = self.session.models.list()
        for model in self.model_list:
            try:
                model.chains
            except AttributeError:
                self.model_list.remove(model)

    def avg_length(self, model_ids, chain_ids):
        count = 0
        for model in self.model_list:
            if model_ids is None:
                pass
            elif model.name not in model_ids:
                continue
            for chain in model.chains:
                if chain_ids is None:
                    pass
                elif chain.chain_id not in chain_ids[model.name]:
                    continue
                for amino_index in range(len(chain.residues)):
                    if chain.residues[amino_index] is None:
                        continue
                    residue = chain.residues[amino_index].atoms
                    if 'CA' != residue.names[1]:
                        continue
                    mid_N_point = residue[0].coord
                    mid_CO_point = residue[2].coord
                    if amino_index != 0 and chain.residues[amino_index - 1] is not None:
                        mid_N_point = (mid_N_point + chain.residues[amino_index - 1].atoms[2].coord) * 0.5
                    if amino_index != len(chain.residues) - 1 and chain.residues[amino_index + 1] is not None:
                        mid_CO_point = (mid_CO_point + chain.residues[amino_index + 1].atoms[0].coord) * 0.5
                    e = np.linalg.norm(mid_N_point - mid_CO_point)
                    self.avg_edge_length += e
                    count += 1
            self.avg_edge_length /= count

    def provide_model(self, model_ids, chain_ids, regular=False):
        amino_count = 0
        amino_skipped_count = 0
        c_alpha_vertex = []
        all_original_vertex = []
        original_c_alpha_vertex = []

        for model in self.model_list:
            if model_ids is None:
                pass
            elif model.name not in model_ids:
                continue
            for chain in model.chains:
                if chain_ids is None:
                    pass
                elif chain.chain_id not in chain_ids[model.name]:
                    continue
                prev_co_cord = None
                for amino_index in range(len(chain.residues)):
                    if chain.residues[amino_index] is None:
                        continue
                    vertex_points = []
                    residue = chain.residues[amino_index].atoms
                    n_cord = residue[0].coord
                    co_cord = residue[2].coord
                    if amino_index != 0 and chain.residues[amino_index - 1] is not None:
                        if regular and prev_co_cord is not None:
                            n_cord = prev_co_cord
                        else:
                            n_cord = (n_cord + chain.residues[amino_index - 1].atoms[2].coord) * 0.5
                    if amino_index != len(chain.residues) - 1 and chain.residues[amino_index + 1] is not None:
                        co_cord = (co_cord + chain.residues[amino_index + 1].atoms[0].coord) * 0.5
                    co_n = n_cord - co_cord
                    norm_co_n = np.linalg.norm(co_n)
                    c_beta_coord = None
                    c_alpha_cord = None
                    if regular:
                        co_n_dir = co_n / norm_co_n
                        co_n = co_n_dir * self.avg_edge_length
                        co_cord = n_cord - co_n
                        norm_co_n = self.avg_edge_length
                        prev_co_cord = co_cord
                    if 'CA' != residue.names[1]:
                        prev_co_cord = None
                        continue
                    if len(residue) == 4:
                        c_alpha_cord = residue[1].coord
                        mid_point_vector = np.random.randint(3, 10, 3)
                        mid_point_vector = np.cross(mid_point_vector, (n_cord - co_cord))
                    elif len(residue) > 4:
                        c_beta_coord = residue[4].coord
                        c_alpha_cord = residue[1].coord
                        co_c_beta = c_beta_coord - co_cord
                        norm_co_c_beta = np.linalg.norm(co_c_beta)
                        move_to_mid_line = (0.5 * norm_co_n - (np.dot(co_n, co_c_beta) / norm_co_n)) * (
                                co_n / norm_co_n)
                        mid_point_vector = c_beta_coord + move_to_mid_line - (co_cord + n_cord) * 0.5
                    mid_point_vector *= np.longdouble(
                        np.sqrt(3, dtype=np.longdouble) * 0.5) * norm_co_n / np.linalg.norm(
                        mid_point_vector)
                    c_beta_coord = (co_cord + n_cord) * 0.5 + mid_point_vector
                    centroid = (c_beta_coord + co_cord + n_cord) / 3
                    direction = np.cross((c_beta_coord - co_cord), (n_cord - co_cord))
                    unit_dir = direction / np.linalg.norm(direction)
                    vec = c_alpha_cord - centroid
                    cos_theta = np.dot(unit_dir, vec) / np.linalg.norm(vec)
                    if cos_theta < 0:
                        unit_dir *= -1
                    H_vector = np.longdouble(
                        np.sqrt(np.longdouble(2) / np.longdouble(3), dtype=np.longdouble)) * norm_co_n * unit_dir
                    h_cord = centroid + H_vector
                    norm_c_beta_n = np.linalg.norm(c_beta_coord - n_cord)
                    norm_co_c_beta = np.linalg.norm(co_cord - c_beta_coord)
                    norm_co_h = np.linalg.norm(co_cord - h_cord)
                    norm_c_beta_h = np.linalg.norm(c_beta_coord - h_cord)
                    norm_n_h = np.linalg.norm(n_cord - h_cord)
                    if len(residue) == 4:
                        original_cb = c_beta_coord
                    else:
                        original_cb = residue[4].coord
                    vertices = [n_cord, co_cord, c_beta_coord, h_cord]
                    original_vertices = np.array([residue[0].coord, residue[2].coord, original_cb, vertices[-1]])
                    edges = np.array([norm_co_n, norm_c_beta_n, norm_co_c_beta, norm_co_h, norm_c_beta_h, norm_n_h])
                    original_edges = np.array([np.linalg.norm(original_vertices[0] - original_vertices[1]),
                                               np.linalg.norm(original_vertices[0] - original_vertices[2]),
                                               np.linalg.norm(original_vertices[1] - original_vertices[2]),
                                               np.linalg.norm(original_vertices[1] - original_vertices[3]),
                                               np.linalg.norm(original_vertices[2] - original_vertices[3]),
                                               np.linalg.norm(original_vertices[0] - original_vertices[3])])
                    face_index = list(combinations(np.arange(amino_count * 4, (amino_count + 1) * 4), 3))
                    self.va.append(vertices)
                    self.ta.extend(face_index)
                    c_alpha_vertex.append(c_alpha_cord)
                    all_original_vertex.extend(original_vertices)
                    original_c_alpha_vertex.append((n_cord + co_cord + c_beta_coord + h_cord) / 4)
                    amino_count += 1

        self.va = np.array(self.va, np.float32)
        self.ta = np.array(self.ta, np.int32)

    def tetrahedron_model(self, pdb_name='1dn3', model_ids=None, chain_ids=None, reg=True):
        if chain_ids is None:
            chains = []
            for model in self.model_list:
                chains.append([i.chain_id for i in model.chains])
            chain_ids = {model.name: chain_id for (model, chain_id) in zip(self.model_list, chains)}
        if model_ids is None:
            model_ids = [model.name for model in self.model_list]

        self.avg_length(model_ids, chain_ids)
        self.provide_model(model_ids, chain_ids, reg)

        va = np.reshape(self.va, (self.va.shape[0] * self.va.shape[1], self.va.shape[2]))
        self.t.set_geometry(va, va, self.ta)

        self.t.vertex_colors = self.model_list[0].atoms.colors
        m0 = Model('m0', self.session)
        m0.add([self.t])
        self.session.models.add([m0])

    def massing(self, seq=False, refinement=2):

        v = []
        if seq:
            v.extend(self.model_list[seq[0]].residues[seq[1]: seq[2]].atoms.coords)
        else:
            seq = (0, 0, len(self.va) - 1)
            for model in self.model_list:
                v.extend(model.atoms.coords)

        mesh = alphashape.alphashape(v, refinement * 0.1)
        inside = trimesh.proximity.ProximityQuery(mesh).signed_distance

        faces = []
        visited = set()
        count = 0
        q = []

        # Create the first tetrahedron
        self.avg_edge_length *= 0.9999
        pt1 = self.va[seq[1]][0]
        pt2 = self.va[seq[1]][0] + (self.va[seq[1]][1] - self.va[seq[1]][0]) * self.avg_edge_length / np.linalg.norm(self.va[seq[1]][1] - self.va[seq[1]][0])
        pt3 = self.va[seq[1]][0] + (self.va[seq[1]][2] - self.va[seq[1]][0]) * self.avg_edge_length / np.linalg.norm(self.va[seq[1]][2] - self.va[seq[1]][0])
        pt4 = self.va[seq[1]][0] + (self.va[seq[1]][3] - self.va[seq[1]][0]) * self.avg_edge_length / np.linalg.norm(self.va[seq[1]][3] - self.va[seq[1]][0])

        print(pt1, pt2, pt3, pt4)
        self.massing_vertices.extend([pt1, pt2, pt3, pt4])
        f = list(combinations(np.arange(count * 4, (count + 1) * 4), 3))
        faces.extend(f)

        pt1, pt2, pt3, pt4 = tuple(pt1), tuple(pt2), tuple(pt3), tuple(pt4)
        q.append({pt1, pt2, pt3, pt4})
        t = tuple(sorted((pt1, pt2, pt3, pt4)))
        visited.add(t)

        depth = 8000
        while q:
            if depth < 0:
                break
            depth -= 1

            # Create new four tetrahedrons
            prev_tetra = list(q.pop())
            combine = combinations(prev_tetra, 3)
            for p in combine:
                for x in prev_tetra:
                    if x not in p:
                        p += (x,)
                        break

                pt1, pt2, pt3, pt4 = p
                pt1, pt2, pt3, pt4 = np.array(pt1), np.array(pt2), np.array(pt3), np.array(pt4)
                p1 = pt2
                p2 = 2 * pt2 - pt1
                p3 = pt2 + pt3 - pt1
                p4 = pt2 + pt4 - pt1
                centroid = (p1 + p2 + p3 + p4) * 0.25

                # Out of boundary
                if inside((centroid,)) < -1:
                    continue

                # Visited or Not
                pt1, pt2, pt3, pt4 = tuple(p1), tuple(p2), tuple(p3), tuple(p4)
                t = tuple(sorted((pt1, pt2, pt3, pt4)))
                if t not in visited:
                    pt1, pt2, pt3, pt4 = np.array(pt1, dtype=np.longdouble), np.array(pt2, dtype=np.longdouble),\
                                         np.array(pt3, dtype=np.longdouble), np.array(pt4, dtype=np.longdouble)

                    self.massing_vertices.extend([pt1, pt2, pt3, pt4])
                    f = list(combinations(np.arange(count * 4, (count + 1) * 4), 3))
                    faces.extend(f)
                    count += 1

                    pt1, pt2, pt3, pt4 = tuple(pt1), tuple(pt2), tuple(pt3), tuple(pt4)
                    q.append({pt1, pt2, pt3, pt4})
                    visited.add(t)
                else:
                    print("END CONDITION")

        self.massing_vertices = np.array(self.massing_vertices)
        faces = np.array(faces, np.int32)
        print(faces)
        print(self.va.shape, self.ta.shape)
        print(self.massing_vertices.shape, faces.shape)

        t = Drawing("t")
        t.set_geometry(self.massing_vertices, self.massing_vertices, faces)
        t.vertex_colors = np.array([[210, 140, 27, 225] for i in range(len(self.massing_vertices))], dtype=np.int8)

        m1 = Model('m1', self.session)
        m1.add([t])
        self.session.models.add([m1])
