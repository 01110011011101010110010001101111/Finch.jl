begin
    x_data = (ex.bodies[1]).body.lhs.tns.bind
    x_val = x_data.val
    yf_lvl = (ex.bodies[1]).body.rhs.tns.bind.lvl
    yf_lvl_ptr = yf_lvl.ptr
    yf_lvl_idx = yf_lvl.idx
    yf_lvl_stop = yf_lvl.shape
    yf_lvl_2 = yf_lvl.lvl
    yf_lvl_2_val = yf_lvl_2.val
    yf_lvl_q = yf_lvl_ptr[1]
    yf_lvl_q_stop = yf_lvl_ptr[1 + 1]
    if yf_lvl_q < yf_lvl_q_stop
        yf_lvl_i1 = yf_lvl_idx[yf_lvl_q_stop - 1]
    else
        yf_lvl_i1 = 0
    end
    phase_stop = min(yf_lvl_i1, yf_lvl_stop)
    if phase_stop >= 1
        i = 1
        if yf_lvl_idx[yf_lvl_q] < 1
            yf_lvl_q = Finch.scansearch(yf_lvl_idx, 1, yf_lvl_q, yf_lvl_q_stop - 1)
        end
        while true
            yf_lvl_i = yf_lvl_idx[yf_lvl_q]
            if yf_lvl_i < phase_stop
                cond = 1 <= -i + yf_lvl_i
                if cond
                    x_val = missing
                end
                yf_lvl_2_val_2 = yf_lvl_2_val[yf_lvl_q]
                x_val = min(x_val, yf_lvl_2_val_2)
                yf_lvl_q += 1
                i = yf_lvl_i + 1
            else
                phase_stop_3 = min(phase_stop, yf_lvl_i)
                if yf_lvl_i == phase_stop_3
                    cond_3 = 1 <= -i + phase_stop_3
                    if cond_3
                        x_val = missing
                    end
                    yf_lvl_2_val_2 = yf_lvl_2_val[yf_lvl_q]
                    x_val = min(x_val, yf_lvl_2_val_2)
                    yf_lvl_q += 1
                else
                    cond_5 = 1 <= 1 + -i + phase_stop_3
                    if cond_5
                        x_val = missing
                    end
                end
                i = phase_stop_3 + 1
                break
            end
        end
    end
    phase_start_3 = max(1, 1 + yf_lvl_i1)
    if yf_lvl_stop >= phase_start_3
        cond_6 = 1 <= 1 + -phase_start_3 + yf_lvl_stop
        if cond_6
            x_val = missing
        end
    end
    result = ()
    x_data.val = x_val
    result
end
