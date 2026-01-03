(set-logic ALL)

(define-fun str.isalpha ((s String)) Bool
  (and (> (str.len s) 0)
       (forall ((i Int))
         (=> (and (<= 0 i) (< i (str.len s)))
             (let ((c (str.to_code (str.at s i))))
               (or (and (>= c 65) (<= c 90))
                   (and (>= c 97) (<= c 122))))))))


(define-fun python.str.at ((s String) (start Int)) String
  (let ((start (ite (< start 0) (+ (str.len s) start) start)))
    (str.substr s start 1)))


; Check if a single character is uppercase (A-Z)
(define-fun char.is_upper ((c String)) Bool
  (and (>= (str.to_code c) 65) (<= (str.to_code c) 90)))

; Check if a single character is lowercase (a-z)
(define-fun char.is_lower ((c String)) Bool
  (and (>= (str.to_code c) 97) (<= (str.to_code c) 122)))

; Convert uppercase char to lowercase
(define-fun char.to_lower ((c String)) String
  (str.from_code (+ (str.to_code c) 32)))

; Convert lowercase char to uppercase
(define-fun char.to_upper ((c String)) String
  (str.from_code (- (str.to_code c) 32)))


(define-fun-rec isupper ((s String)) Bool
  (ite (= s "")
       true
       (and (char.is_upper (str.at s 0))
            (isupper (str.substr s 1 (- (str.len s) 1))))))

(declare-const x String)
(assert (and (= (str.len x) 5) (forall ((str_pos_1 Int)) (=> (and (>= str_pos_1 0) (< str_pos_1 (str.len x))) (or (isupper (python.str.at x str_pos_1)) (not (str.isalpha (python.str.at x str_pos_1))))))))
(check-sat)
(get-model)