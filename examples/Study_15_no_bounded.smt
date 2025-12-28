(set-logic ALL)

(declare-datatypes ((List 1))
    ((par (T) ((cons (head T) (tail (List T))) (nil)))))


(define-fun list.adjust_index ((idx Int) (len Int)) Int
  (ite (< idx 0)
       (+ len idx)
       idx))


(define-fun-rec list.length.int ((l (List Int))) Int
  (ite (= l (as nil (List Int)))
       0
       (+ 1 (list.length.int (tail l)))))


(define-fun-rec list.get.int ((l (List Int)) (idx Int)) Int
  (ite (< idx 0)
       (list.get.int l (+ (list.length.int l) idx))
  (ite (= idx 0)
       (head l)
       (list.get.int (tail l) (- idx 1)))))


(define-fun-rec list.reverse.int ((l (List Int)) (acc (List Int))) (List Int)
  (ite (= l (as nil (List Int)))
       acc
       (list.reverse.int (tail l) (cons (head l) acc))))


(define-fun list.valid_index.int ((l (List Int)) (idx Int)) Bool
  (and (>= idx 0) (< idx (list.length.int l))))

(define-fun-rec list.slice.int.helper ((l (List Int)) (curr Int) (stop Int) (step Int) (result (List Int))) (List Int)
  (ite (or (and (> step 0) (>= curr stop))
           (and (< step 0) (<= curr stop))
           (not (list.valid_index.int l curr)))
       result
       (let ((new_result (cons (list.get.int l curr) result)))
         (list.slice.int.helper l (+ curr step) stop step new_result))))

(define-fun list.slice.int ((l (List Int)) (start Int) (stop Int) (step Int)) (List Int)
  (let ((len (list.length.int l)))
    (ite (= step 0)
         (as nil (List Int))
         (let ((adj_start (list.adjust_index start len))
               (adj_stop (list.adjust_index stop len)))
           (ite (> step 0)
                (list.reverse.int
                  (list.slice.int.helper l adj_start adj_stop step (as nil (List Int)))
                  (as nil (List Int)))
                (let ((real_start (- len 1 adj_start))
                      (real_stop (- len 1 adj_stop)))
                  (list.reverse.int
                    (list.slice.int.helper l real_start real_stop (ite (< step 0) (- 0 step) step) (as nil (List Int)))
                    (as nil (List Int)))))))))


(define-fun-rec list.sum.int ((l (List Int))) Int
  (ite (= l (as nil (List Int)))
       0
       (+ (head l) (list.sum.int (tail l)))))

(declare-const x (List Int))
(assert (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (= (list.sum.int (list.slice.int x 0 -1 1)) 0) (= (list.sum.int (list.slice.int x 0 1 1)) 1)) (= (list.sum.int (list.slice.int x 0 2 1)) 3)) (= (list.sum.int (list.slice.int x 0 3 1)) 7)) (= (list.sum.int (list.slice.int x 0 4 1)) 15)) (= (list.sum.int (list.slice.int x 0 5 1)) 31)) (= (list.sum.int (list.slice.int x 0 6 1)) 63)) (= (list.sum.int (list.slice.int x 0 7 1)) 127)) (= (list.sum.int (list.slice.int x 0 8 1)) 255)) (= (list.sum.int (list.slice.int x 0 9 1)) 511)) (= (list.sum.int (list.slice.int x 0 10 1)) 1023)) (= (list.sum.int (list.slice.int x 0 11 1)) 2047)) (= (list.sum.int (list.slice.int x 0 12 1)) 4095)) (= (list.sum.int (list.slice.int x 0 13 1)) 8191)) (= (list.sum.int (list.slice.int x 0 14 1)) 16383)) (= (list.sum.int (list.slice.int x 0 15 1)) 32767)) (= (list.sum.int (list.slice.int x 0 16 1)) 65535)) (= (list.sum.int (list.slice.int x 0 17 1)) 131071)) (= (l...
(check-sat)
(get-model)