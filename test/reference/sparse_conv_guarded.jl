begin
    C_lvl = ex.body.body.lhs.tns.tns.lvl
    C_lvl_2 = C_lvl.lvl
    C_lvl_2_val = 0.0
    A_lvl = ((ex.body.body.rhs.args[1]).args[1]).tns.tns.lvl
    A_lvl_2 = A_lvl.lvl
    A_lvl_2_val = 0.0
    A_lvl_3 = ((ex.body.body.rhs.args[2]).args[1]).tns.tns.lvl
    A_lvl_4 = A_lvl_3.lvl
    A_lvl_4_val = 0.0
    F_lvl = ((ex.body.body.rhs.args[3]).args[1]).tns.tns.lvl
    F_lvl_2 = F_lvl.lvl
    F_lvl_2_val = 0
    C_lvl_qos_fill = 0
    C_lvl_qos_stop = 0
    (Finch.resize_if_smaller!)(C_lvl.pos, 1 + 1)
    (Finch.fill_range!)(C_lvl.pos, 0, 1 + 1, 1 + 1)
    C_lvl_qos = C_lvl_qos_fill + 1
    A_lvl_q = A_lvl.pos[1]
    A_lvl_q_stop = A_lvl.pos[1 + 1]
    A_lvl_i = if A_lvl_q < A_lvl_q_stop
            A_lvl.idx[A_lvl_q]
        else
            1
        end
    A_lvl_i1 = if A_lvl_q < A_lvl_q_stop
            A_lvl.idx[A_lvl_q_stop - 1]
        else
            0
        end
    i = 1
    i_start = i
    phase_stop = (min)(A_lvl_i1, A_lvl.I)
    if phase_stop >= i_start
        i = i
        i = i_start
        while A_lvl_q + 1 < A_lvl_q_stop && A_lvl.idx[A_lvl_q] < i_start
            A_lvl_q += 1
        end
        while i <= phase_stop
            i_start_2 = i
            A_lvl_i = A_lvl.idx[A_lvl_q]
            phase_stop_2 = (min)(A_lvl_i, phase_stop)
            i_2 = i
            if A_lvl_i == phase_stop_2
                A_lvl_2_val = A_lvl_2.val[A_lvl_q]
                i_3 = phase_stop_2
                if C_lvl_qos > C_lvl_qos_stop
                    C_lvl_qos_stop = max(C_lvl_qos_stop << 1, 1)
                    (Finch.resize_if_smaller!)(C_lvl.idx, C_lvl_qos_stop)
                    resize_if_smaller!(C_lvl_2.val, C_lvl_qos_stop)
                    fill_range!(C_lvl_2.val, 0.0, C_lvl_qos, C_lvl_qos_stop)
                end
                C_lvl_2_dirty = false
                delta = (+)(-3, i_3)
                C_lvl_2_val = C_lvl_2.val[C_lvl_qos]
                j_start = (min)(1, (+)(1, delta))
                j_stop = (max)(F_lvl.I, (+)(delta, A_lvl_3.I))
                j = j_start
                j_start_2 = j
                phase_start = (max)(j_start_2, (+)(delta, j_start_2, (-)(delta)))
                phase_stop_3 = (min)(0, delta, j_stop)
                if phase_stop_3 >= phase_start
                    j = j
                    j = phase_stop_3 + 1
                end
                j_start_2 = j
                phase_start_2 = (max)(j_start_2, (+)(delta, j_start_2, (-)(delta)))
                phase_stop_4 = (min)(0, (+)(delta, A_lvl_3.I), j_stop)
                if phase_stop_4 >= phase_start_2
                    j_2 = j
                    A_lvl_3_q = A_lvl_3.pos[1]
                    A_lvl_3_q_stop = A_lvl_3.pos[1 + 1]
                    A_lvl_3_i = if A_lvl_3_q < A_lvl_3_q_stop
                            A_lvl_3.idx[A_lvl_3_q]
                        else
                            1
                        end
                    A_lvl_3_i1 = if A_lvl_3_q < A_lvl_3_q_stop
                            A_lvl_3.idx[A_lvl_3_q_stop - 1]
                        else
                            0
                        end
                    j = phase_start_2
                    j_start_3 = j
                    phase_start_3 = (max)(j_start_3, (+)(delta, (-)(delta), j_start_3))
                    phase_stop_5 = (min)(phase_stop_4, (+)(delta, A_lvl_3_i1))
                    if phase_stop_5 >= phase_start_3
                        j_3 = j
                        j = phase_stop_5 + 1
                    end
                    j_start_3 = j
                    phase_start_4 = (max)(j_start_3, (+)(delta, (-)(delta), j_start_3))
                    phase_stop_6 = (min)(phase_stop_4, (+)(delta, (-)(delta), phase_stop_4))
                    if phase_stop_6 >= phase_start_4
                        j_4 = j
                        j = phase_stop_6 + 1
                    end
                    j = phase_stop_4 + 1
                end
                j_start_2 = j
                phase_start_5 = (max)(j_start_2, (+)(delta, j_start_2, (-)(delta)))
                phase_stop_7 = (min)(0, j_stop, (+)(delta, (-)(delta), j_stop))
                if phase_stop_7 >= phase_start_5
                    j_5 = j
                    j = phase_stop_7 + 1
                end
                j_start_2 = j
                phase_start_6 = (max)(j_start_2, (+)(delta, j_start_2, (-)(delta)))
                phase_stop_8 = (min)(delta, F_lvl.I, j_stop)
                if phase_stop_8 >= phase_start_6
                    j_6 = j
                    j = phase_stop_8 + 1
                end
                j_start_2 = j
                phase_start_7 = (max)(j_start_2, (+)(delta, j_start_2, (-)(delta)))
                phase_stop_9 = (min)(F_lvl.I, (+)(delta, A_lvl_3.I), j_stop)
                if phase_stop_9 >= phase_start_7
                    j_7 = j
                    A_lvl_3_q = A_lvl_3.pos[1]
                    A_lvl_3_q_stop = A_lvl_3.pos[1 + 1]
                    A_lvl_3_i = if A_lvl_3_q < A_lvl_3_q_stop
                            A_lvl_3.idx[A_lvl_3_q]
                        else
                            1
                        end
                    A_lvl_3_i1 = if A_lvl_3_q < A_lvl_3_q_stop
                            A_lvl_3.idx[A_lvl_3_q_stop - 1]
                        else
                            0
                        end
                    j = phase_start_7
                    j_start_4 = j
                    phase_start_8 = (max)(j_start_4, (+)(delta, (-)(delta), j_start_4))
                    phase_stop_10 = (min)((+)(delta, A_lvl_3_i1), phase_stop_9)
                    if phase_stop_10 >= phase_start_8
                        j_8 = j
                        j = phase_start_8
                        while A_lvl_3_q + 1 < A_lvl_3_q_stop && A_lvl_3.idx[A_lvl_3_q] < (+)(phase_start_8, (-)(delta))
                            A_lvl_3_q += 1
                        end
                        while j <= phase_stop_10
                            j_start_5 = j
                            A_lvl_3_i = A_lvl_3.idx[A_lvl_3_q]
                            phase_start_9 = (max)(j_start_5, (+)(delta, (-)(delta), j_start_5))
                            phase_stop_11 = (min)(phase_stop_10, (+)(delta, A_lvl_3_i))
                            if phase_stop_11 >= phase_start_9
                                j_9 = j
                                if A_lvl_3_i == (+)(phase_stop_11, (-)(delta))
                                    A_lvl_4_val = A_lvl_4.val[A_lvl_3_q]
                                    j_10 = phase_stop_11
                                    F_lvl_q = (1 - 1) * F_lvl.I + j_10
                                    F_lvl_2_val = F_lvl_2.val[F_lvl_q]
                                    C_lvl_2_dirty = true
                                    C_lvl_2_dirty = true
                                    C_lvl_2_val = (+)((*)((!=)(A_lvl_2_val, 0), (coalesce)(F_lvl_2_val, 0), (coalesce)(A_lvl_4_val, 0)), C_lvl_2_val)
                                    A_lvl_3_q += 1
                                else
                                end
                                j = phase_stop_11 + 1
                            end
                        end
                        j = phase_stop_10 + 1
                    end
                    j_start_4 = j
                    phase_start_10 = (max)(j_start_4, (+)(delta, (-)(delta), j_start_4))
                    phase_stop_12 = (min)(phase_stop_9, (+)(delta, (-)(delta), phase_stop_9))
                    if phase_stop_12 >= phase_start_10
                        j_11 = j
                        j = phase_stop_12 + 1
                    end
                    j = phase_stop_9 + 1
                end
                j_start_2 = j
                phase_start_11 = (max)(j_start_2, (+)(delta, j_start_2, (-)(delta)))
                phase_stop_13 = (min)(F_lvl.I, j_stop, (+)(delta, (-)(delta), j_stop))
                if phase_stop_13 >= phase_start_11
                    j_12 = j
                    j = phase_stop_13 + 1
                end
                j_start_2 = j
                phase_start_12 = (max)(j_start_2, (+)(delta, j_start_2, (-)(delta)))
                phase_stop_14 = (min)(delta, j_stop)
                if phase_stop_14 >= phase_start_12
                    j_13 = j
                    j = phase_stop_14 + 1
                end
                j_start_2 = j
                phase_start_13 = (max)(j_start_2, (+)(delta, j_start_2, (-)(delta)))
                phase_stop_15 = (min)((+)(delta, A_lvl_3.I), j_stop)
                if phase_stop_15 >= phase_start_13
                    j_14 = j
                    A_lvl_3_q = A_lvl_3.pos[1]
                    A_lvl_3_q_stop = A_lvl_3.pos[1 + 1]
                    A_lvl_3_i = if A_lvl_3_q < A_lvl_3_q_stop
                            A_lvl_3.idx[A_lvl_3_q]
                        else
                            1
                        end
                    A_lvl_3_i1 = if A_lvl_3_q < A_lvl_3_q_stop
                            A_lvl_3.idx[A_lvl_3_q_stop - 1]
                        else
                            0
                        end
                    j = phase_start_13
                    j_start_6 = j
                    phase_start_14 = (max)(j_start_6, (+)(delta, (-)(delta), j_start_6))
                    phase_stop_16 = (min)((+)(delta, A_lvl_3_i1), phase_stop_15)
                    if phase_stop_16 >= phase_start_14
                        j_15 = j
                        j = phase_stop_16 + 1
                    end
                    j_start_6 = j
                    phase_start_15 = (max)(j_start_6, (+)(delta, (-)(delta), j_start_6))
                    phase_stop_17 = (min)(phase_stop_15, (+)(delta, (-)(delta), phase_stop_15))
                    if phase_stop_17 >= phase_start_15
                        j_16 = j
                        j = phase_stop_17 + 1
                    end
                    j = phase_stop_15 + 1
                end
                j_start_2 = j
                phase_start_16 = (max)(j_start_2, (+)(delta, j_start_2, (-)(delta)))
                phase_stop_18 = (min)(j_stop, (+)(delta, (-)(delta), j_stop))
                if phase_stop_18 >= phase_start_16
                    j_17 = j
                    j = phase_stop_18 + 1
                end
                C_lvl_2.val[C_lvl_qos] = C_lvl_2_val
                if C_lvl_2_dirty
                    C_lvl_dirty = true
                    C_lvl.idx[C_lvl_qos] = i_3
                    C_lvl_qos += 1
                end
                A_lvl_q += 1
            else
            end
            i = phase_stop_2 + 1
        end
        i = phase_stop + 1
    end
    i_start = i
    if A_lvl.I >= i_start
        i_4 = i
        i = A_lvl.I + 1
    end
    C_lvl.pos[1 + 1] = (C_lvl_qos - C_lvl_qos_fill) - 1
    C_lvl_qos_fill = C_lvl_qos - 1
    for p = 2:1 + 1
        C_lvl.pos[p] += C_lvl.pos[p - 1]
    end
    qos_stop = C_lvl.pos[1 + 1] - 1
    resize!(C_lvl.pos, 1 + 1)
    qos = C_lvl.pos[end] - 1
    resize!(C_lvl.idx, qos)
    resize!(C_lvl_2.val, qos)
    (C = Fiber((Finch.SparseListLevel){Int64}(A_lvl.I, C_lvl.pos, C_lvl.idx, C_lvl_2), (Environment)(; )),)
end
