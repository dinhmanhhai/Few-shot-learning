"""
Module cho episode training và evaluation với Relation Network
"""
import torch
import torch.nn.functional as F
import numpy as np
from models.backbone import RelationNetworkModel

def run_episode_with_detailed_evaluation(model, dataset, config, use_augmentation=True, include_validation=False):
    """
    Chạy một episode với đánh giá chi tiết sử dụng Relation Network
    """
    model.eval()
    
    N_WAY = config['N_WAY']
    K_SHOT = config['K_SHOT']
    Q_QUERY = config['Q_QUERY']
    Q_VALID = config['Q_VALID']
    USE_VALIDATION = config['USE_VALIDATION']
    DEVICE = 'cuda' if torch.cuda.is_available() and config['USE_CUDA'] else 'cpu'
    
    if include_validation and USE_VALIDATION:
        support_set, query_set, valid_set = dataset.sample_episode(N_WAY, K_SHOT, Q_QUERY, Q_VALID, use_augmentation)
        support_imgs, support_labels = zip(*support_set)
        query_imgs, query_labels = zip(*query_set)
        valid_imgs, valid_labels = zip(*valid_set)
        
        # Lấy tên class thực tế được sử dụng trong episode này
        episode_class_names = dataset.selected_class_names

        support_imgs = torch.stack(support_imgs).to(DEVICE)
        support_labels = torch.tensor(support_labels).to(DEVICE)
        query_imgs = torch.stack(query_imgs).to(DEVICE)
        query_labels = torch.tensor(query_labels).to(DEVICE)
        valid_imgs = torch.stack(valid_imgs).to(DEVICE)
        valid_labels = torch.tensor(valid_labels).to(DEVICE)

        with torch.no_grad():
            # Query evaluation
            query_class_scores = model.compute_class_scores(support_imgs, support_labels, query_imgs, N_WAY)
            query_log_p_y = F.log_softmax(query_class_scores, dim=1)
            query_loss = F.nll_loss(query_log_p_y, query_labels)
            query_acc = (query_log_p_y.argmax(1) == query_labels).float().mean()
            query_predictions = query_log_p_y.argmax(1).cpu().numpy()
            query_targets = query_labels.cpu().numpy()
            
            # Validation evaluation
            valid_class_scores = model.compute_class_scores(support_imgs, support_labels, valid_imgs, N_WAY)
            valid_log_p_y = F.log_softmax(valid_class_scores, dim=1)
            valid_loss = F.nll_loss(valid_log_p_y, valid_labels)
            valid_acc = (valid_log_p_y.argmax(1) == valid_labels).float().mean()
            valid_predictions = valid_log_p_y.argmax(1).cpu().numpy()
            valid_targets = valid_labels.cpu().numpy()

        return {
            'query_loss': query_loss.item(),
            'query_acc': query_acc.item(),
            'query_predictions': query_predictions,
            'query_targets': query_targets,
            'valid_loss': valid_loss.item(),
            'valid_acc': valid_acc.item(),
            'valid_predictions': valid_predictions,
            'valid_targets': valid_targets,
            'episode_class_names': episode_class_names
        }
    else:
        support_set, query_set, _ = dataset.sample_episode(N_WAY, K_SHOT, Q_QUERY, 0, use_augmentation)
        support_imgs, support_labels = zip(*support_set)
        query_imgs, query_labels = zip(*query_set)
        
        # Lấy tên class thực tế được sử dụng trong episode này
        episode_class_names = dataset.selected_class_names

        support_imgs = torch.stack(support_imgs).to(DEVICE)
        support_labels = torch.tensor(support_labels).to(DEVICE)
        query_imgs = torch.stack(query_imgs).to(DEVICE)
        query_labels = torch.tensor(query_labels).to(DEVICE)

        with torch.no_grad():
            # Compute class scores using Relation Network
            class_scores = model.compute_class_scores(support_imgs, support_labels, query_imgs, N_WAY)
            log_p_y = F.log_softmax(class_scores, dim=1)
            loss = F.nll_loss(log_p_y, query_labels)
            acc = (log_p_y.argmax(1) == query_labels).float().mean()
            predictions = log_p_y.argmax(1).cpu().numpy()
            targets = query_labels.cpu().numpy()

        return {
            'query_loss': loss.item(),
            'query_acc': acc.item(),
            'query_predictions': predictions,
            'query_targets': targets,
            'valid_loss': None,
            'valid_acc': None,
            'valid_predictions': None,
            'valid_targets': None,
            'episode_class_names': episode_class_names
        }

def run_multiple_episodes_with_detailed_evaluation(model, dataset, config, num_episodes, use_augmentation=True, include_validation=False):
    """
    Chạy nhiều episodes với đánh giá chi tiết sử dụng Relation Network
    """
    query_losses = []
    query_accuracies = []
    valid_losses = []
    valid_accuracies = []
    
    # Thu thập predictions và targets cho đánh giá tổng hợp
    all_query_predictions = []
    all_query_targets = []
    all_valid_predictions = []
    all_valid_targets = []
    
    # Thu thập tên class được sử dụng trong các episodes
    all_episode_class_names = []
    
    print(f"🔄 Đang chạy {num_episodes} episodes với Relation Network...")
    
    for episode in range(num_episodes):
        results = run_episode_with_detailed_evaluation(model, dataset, config, use_augmentation, include_validation)
        query_losses.append(results['query_loss'])
        query_accuracies.append(results['query_acc'])
        
        # Thu thập predictions và targets
        all_query_predictions.extend(results['query_predictions'])
        all_query_targets.extend(results['query_targets'])
        
        # Thu thập tên class
        all_episode_class_names.append(results['episode_class_names'])
        
        if results['valid_loss'] is not None:
            valid_losses.append(results['valid_loss'])
            valid_accuracies.append(results['valid_acc'])
            all_valid_predictions.extend(results['valid_predictions'])
            all_valid_targets.extend(results['valid_targets'])
        
        # In tiến độ theo cấu hình
        if config['DISPLAY_PROGRESS'] and ((episode + 1) % 5 == 0 or episode == 0):
            if results['valid_loss'] is not None:
                print(f"   Episode {episode+1}/{num_episodes}: Q_Loss={results['query_loss']:.4f}, Q_Acc={results['query_acc']:.4f}, V_Loss={results['valid_loss']:.4f}, V_Acc={results['valid_acc']:.4f}")
                print(f"      Classes: {results['episode_class_names']}")
            else:
                print(f"   Episode {episode+1}/{num_episodes}: Loss={results['query_loss']:.4f}, Acc={results['query_acc']:.4f}")
                print(f"      Classes: {results['episode_class_names']}")
    
    # Tính thống kê query
    avg_query_loss = np.mean(query_losses)
    avg_query_acc = np.mean(query_accuracies)
    std_query_loss = np.std(query_losses)
    std_query_acc = np.std(query_accuracies)
    min_query_acc = np.min(query_accuracies)
    max_query_acc = np.max(query_accuracies)
    
    result = {
        'query_losses': query_losses,
        'query_accuracies': query_accuracies,
        'avg_query_loss': avg_query_loss,
        'avg_query_acc': avg_query_acc,
        'std_query_loss': std_query_loss,
        'std_query_acc': std_query_acc,
        'min_query_acc': min_query_acc,
        'max_query_acc': max_query_acc,
        'all_query_predictions': np.array(all_query_predictions),
        'all_query_targets': np.array(all_query_targets),
        'all_episode_class_names': all_episode_class_names
    }
    
    # Tính thống kê validation nếu có
    if valid_losses:
        avg_valid_loss = np.mean(valid_losses)
        avg_valid_acc = np.mean(valid_accuracies)
        std_valid_loss = np.std(valid_losses)
        std_valid_acc = np.std(valid_accuracies)
        min_valid_acc = np.min(valid_accuracies)
        max_valid_acc = np.max(valid_accuracies)
        
        result.update({
            'valid_losses': valid_losses,
            'valid_accuracies': valid_accuracies,
            'avg_valid_loss': avg_valid_loss,
            'avg_valid_acc': avg_valid_acc,
            'std_valid_loss': std_valid_loss,
            'std_valid_acc': std_valid_acc,
            'min_valid_acc': min_valid_acc,
            'max_valid_acc': max_valid_acc,
            'all_valid_predictions': np.array(all_valid_predictions),
            'all_valid_targets': np.array(all_valid_targets)
        })
    
    return result
